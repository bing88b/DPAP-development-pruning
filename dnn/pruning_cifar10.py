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
from maskup import *
from datetime import datetime


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', default='/data0/datasets/CIFAR10/',type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='vgg16_bn', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
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
#compress rate
parser.add_argument('--rate', type=float, default=0.7, help='compress rate of model')
parser.add_argument('--layer_begin', type=int, default=0,  help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=90,  help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=3,  help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1,  help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')


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

# th={}
# convtra = {}
# wwfc={}
# epoch_tra = {}
# size={}
# layer=46
# for i in range(layer):
#     if int((i-1)/15)==0:
#         size[i]=16
#     elif int((i-1)/15)==1:
#         size[i]=32
#     elif int((i-1)/15)==2:
#         size[i]=64
    
#     if (i%3)==0:
#         pass
#     else:
#         th[i]=torch.zeros((args.batch_size,size[i]),device=device)
#         convtra[i] = torch.zeros(size[i],device=device)
#         wwfc[i]=torch.zeros(size[i],size[i-1],device=device)
#         epoch_tra[i] = torch.zeros((size[i]),device=device)

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

def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, '{}_{}.txt'.format(datetime.now().strftime("%Y%m%d-%H%M%S"),'civggbn8-c-0.6-13-fc-0.8_15')), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

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

        if epoch>4:
            m.model = net
            m.init_mask(wwfc,convtra,epoch)
            m.do_mask()
            cc=m.if_zero()
            net = m.model

        # for i in range(layer):
        #     if (i%3)!=0:
        #         wconv = torch.sum(wwfc[i], dim=1)
        #         wconv=unit_tensor(wconv)
        #         traconv=unit_tensor(epoch_tra[i])
        #         convtra[i]=wconv*traconv

        # evaluate on validation set
        # #val_acc_1,   val_los_1   = validate(test_loader, net, criterion, log)
        # if ((epoch>5)and(epoch % args.epoch_prune ==0 or epoch == args.epochs-1)):
        #     m.model = net
        #     #m.if_zero()
        #     m.init_mask(convtra,epoch)
        #     m.do_mask()
        #     m.if_zero()
        #     net = m.model 
        #     if args.use_cuda:
        #         net = net.cuda()  
            
        val_acc_2,   val_los_2   = validate(test_loader, net, criterion, log)
        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)
        if epoch>4:
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

        # for i in range(layer):
        #     spike[i]=torch.sum(torch.sum(spike[i],dim=2),dim=2)
        #     if (i%3)!=0:
        #         #print(spike[i].size(),th[i].size())
        #         post1 = (spike[i] * (spike[i] - th[i]))
        #         hebb1 = torch.mm(post1.T, spike[i - 1])
        #         wwfc[i] = wwfc[i] + hebb1
        #         th[i] = torch.div(th[i] * (NUM - 1) + spike[i], NUM)
        #         cs=torch.sum(spike[i],dim=0)
        #         epoch_tra[i] = epoch_tra[i] + cs
        # measure accuracy and record loss
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


# class Mask:
#     def __init__(self,model):
#         self.model_size = {}
#         self.model_length = {}
#         self.compress_rate = {}
#         self.mat = {}
#         self.model = model
#         self.mask_index = []
        
    
#     def get_codebook(self, weight_torch,compress_rate,length):
#         weight_vec = weight_torch.view(length)
#         weight_np = weight_vec.cpu().numpy()
    
#         weight_abs = np.abs(weight_np)
#         weight_sort = np.sort(weight_abs)
        
#         threshold = weight_sort[int (length * (1-compress_rate) )]
#         weight_np [weight_np <= -threshold  ] = 1
#         weight_np [weight_np >= threshold  ] = 1
#         weight_np [weight_np !=1  ] = 0
        
#         print("codebook done")
#         return weight_np

#     def get_filter_codebook(self, weight_torch,compress_rate,length):
#         codebook = np.ones(length)
#         if len( weight_torch.size())==4:
#             filter_pruned_num = int(weight_torch.size()[0]*(1-compress_rate))
#             weight_vec = weight_torch.view(weight_torch.size()[0],-1)
#             norm2 = torch.norm(weight_vec,2,1)
#             norm2_np = norm2.cpu().numpy()
#             filter_index = norm2_np.argsort()[:filter_pruned_num]
# #            norm1_sort = np.sort(norm1_np)
# #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
#             kernel_length = weight_torch.size()[1] *weight_torch.size()[2] *weight_torch.size()[3]
#             for x in range(0,len(filter_index)):
#                 codebook [filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] = 0

#             #print("filter codebook done")
#         else:
#             pass
#         return codebook
    
#     def convert2tensor(self,x):
#         x = torch.FloatTensor(x)
#         return x
    
#     def init_length(self):
#         for index, item in enumerate(self.model.parameters()):
#             self.model_size [index] = item.size()
        
#         for index1 in self.model_size:
#             for index2 in range(0,len(self.model_size[index1])):
#                 if index2 ==0:
#                     self.model_length[index1] = self.model_size[index1][0]
#                 else:
#                     self.model_length[index1] *= self.model_size[index1][index2]
                    
#     def init_rate(self, layer_rate):
#         for index, item in enumerate(self.model.parameters()):
#             self.compress_rate [index] = 1
#         for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
#             self.compress_rate[key]= layer_rate
#         #different setting for  different architecture
#         if args.arch == 'resnet20':
#             last_index = 57
#         elif args.arch == 'resnet32':
#             last_index = 93
#         elif args.arch == 'resnet56':
#             last_index = 165
#         elif args.arch == 'resnet110':
#             last_index = 327
#         self.mask_index =  [x for x in range (0,last_index,3)]
# #        self.mask_index =  [x for x in range (0,330,3)]
        
#     def init_mask(self,layer_rate):
#         self.init_rate(layer_rate)
#         for index, item in enumerate(self.model.parameters()):
#             if(index in self.mask_index):
#                 self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],self.model_length[index] )
#                 self.mat[index] = self.convert2tensor(self.mat[index])
#                 if args.use_cuda:
#                     self.mat[index] = self.mat[index].cuda()
#         print("mask Ready")

#     def do_mask(self):
#         for index, item in enumerate(self.model.parameters()):
#             if(index in self.mask_index):
#                 a = item.data.view(self.model_length[index])
#                 b = a * self.mat[index]
#                 item.data = b.view(self.model_size[index])
#         print("mask Done")

#     def if_zero(self):
#         for index, item in enumerate(self.model.parameters()):
# #            if(index in self.mask_index):
#             if(index in self.mask_index):
#                 a = item.data.view(self.model_length[index])
#                 b = a.cpu().numpy()
                
#                 print("number of nonzero weight is %d, zero is %d" %( np.count_nonzero(b),len(b)- np.count_nonzero(b)))
        