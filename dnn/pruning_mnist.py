from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import numpy as np
from maskupmist import *
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', default='/data0/datasets/CIFAR10/',type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, default='mnist',choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='vgg_mnistbn', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=300, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[1,60,120,160 ], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[10,0.2,0.2,0.2], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=2, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.manualSeed is None:
    args.manualSeed = 5893 #random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
torch.backends.cudnn.deterministic = True

NUM = 0
th={}
convtra = {}
wwfc={}
epoch_tra = {}
for i in range(1,len(convlayer)):
    index=convlayer[i]
    th[index]=torch.zeros((batch,size[i]),device=device)
    convtra[index] = torch.zeros(size[i],device=device)
    wwfc[index]=torch.zeros(size[i],size[i-1],device=device)
    epoch_tra[index] = torch.zeros((size[i]),device=device)
for i in range(1,len(fclayer)):
    index=fclayer[i]
    th[index]=torch.zeros((batch,fcsize[i]),device=device)
    convtra[index]=torch.zeros(fcsize[i],device=device)
    wwfc[index]=torch.zeros(fcsize[i],fcsize[i-1],device=device)
    epoch_tra[index] = torch.zeros(fcsize[i],device=device)

class MnistNet(nn.Module):
    """Small network designed for Mnist debugging
    """
    def __init__(self, pretrained=False):
        assert not pretrained, f"{self.__class__.__name__} does not support pretrained weights"
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1,2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1,2)
        self.fc1 = nn.Linear(7*7*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.fc2.is_classifier = True

    def forward(self, x):
        spikes=[]
        spikes.append(x.detach())
        x=self.conv1(x)
        spikes.append(x.detach())
        x = F.relu(x)
        spikes.append(x.detach())
        x = F.max_pool2d(x, 2, 2)
        spikes.append(x.detach())
        x=self.conv2(x)
        spikes.append(x.detach())
        x = F.relu(x)
        spikes.append(x.detach())
        x = F.max_pool2d(x, 2, 2)
        spikes.append(x.detach())
        x = x.view(-1, 7*7*50)
        spikes.append(x.detach())
        x=self.fc1(x)
        spikes.append(x.detach())
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1),spikes


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'MINSTwww{}_{}_log_seed_{}.txt'.format(datetime.now().strftime("%Y%m%d-%H%M%S"),'c-0.525-10-fc-0.75_12',args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    data_path = '/data1/data_hanbing/stbp3/BP-for-SpikingNN-master/raw/'  # todo: input your data path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = dset.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_set = dset.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_classes=10

    # print_log("=> creating model '{}'".format(args.arch), log)
    # # Init model, criterion, and optimizer
    # net = models.__dict__[args.arch](num_classes)
    # print_log("=> network :\n {}".format(net), log)
    net=MnistNet()

    #net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],weight_decay=state['decay'], nesterov=True)


    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = checkpoint['state_dict']
                
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print ('function took %0.3f ms' % ((time2-time1)*1000.0))
        return
    print(net)
    m=Mask(net)  
    m.init_length()
    m.model = net
    #m.if_zero()
    # m.init_mask(comp_rate)
    # m.if_zero()
    # m.do_mask()
    # net = m.model
    # m.if_zero()
    # if args.use_cuda:
    #     net = net.cuda()    
    
    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    NUM = 0
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los,epoch_tra,wwfc,N = train(train_loader, net, criterion, optimizer, epoch,log,NUM)
        NUM = N

        for i in range(1,len(convlayer)):
            index=convlayer[i]
            wconv = torch.sum(wwfc[index], dim=1)
            wconv=unit_tensor(wconv)
            traconv=unit_tensor(epoch_tra[index])
            convtra[index]=wconv*traconv
        for i in range(1, len(fclayer)):
            index=fclayer[i]
            wfc = torch.sum(wwfc[index], dim=1)
            wfc=unit_tensor(wfc)
            trafc=unit_tensor(epoch_tra[index])
            convtra[index]=wfc*trafc

        if epoch>-1:
            m.model = net
            m.init_mask(wwfc,convtra,epoch)
            m.do_mask()
            cc=m.if_zero()
            net = m.model
 
            
        val_acc_2,   val_los_2   = validate(test_loader, net, criterion, log)
        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)
        if epoch>-1:
            print_log('*** epoch: {0} (pruning {1})'.format(epoch, cc), log)
  
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

    log.close()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log,NUM):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output,spike = model(input_var)
        loss = criterion(output, target_var)

        NUM = NUM + 1
        #spike[-1]=torch.sum(torch.sum(spike[-1],dim=2),dim=2)
        for i in range(1,len(convlayer)):
            index=convlayer[i]
            #print(index,spike[index-1].size(),th[index].size())
            spike[index]=torch.sum(torch.sum(spike[index],dim=2),dim=2)
            spike[index-1]=torch.sum(torch.sum(spike[index-1],dim=2),dim=2)
            post1 = (spike[index] * (spike[index] - th[index]))
            hebb1 = torch.mm(post1.T, spike[index-1])#,spike[convlayer[i - 1]]
            wwfc[index] = wwfc[index] + hebb1
            th[index] = torch.div(th[index] * (NUM - 1) + spike[index], NUM)
            cs=torch.sum(spike[index],dim=0)
            epoch_tra[index] = epoch_tra[index] + cs

        for i in range(1,len(fclayer)):
            index = fclayer[i]
            post1 = (spike[index] * (spike[index] - th[index]))
            hebb1 = torch.mm(post1.T, spike[index-1])#spike[fclayer[i - 1]]
            wwfc[index] = wwfc[index] + hebb1
            th[index] = torch.div(th[index] * (NUM - 1) +spike[index], NUM)
            cs=torch.sum(spike[index],dim=0)
            epoch_tra[index] = epoch_tra[index] + cs

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size()[0])
        top1.update(prec1.item(), input.size()[0])
        top5.update(prec5.item(), input.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, losses.avg,epoch_tra,wwfc,NUM

def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output,spike = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size()[0])
        top1.update(prec1.item(), input.size()[0])
        top5.update(prec5.item(), input.size()[0])

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

