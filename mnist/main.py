# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018

@author: yjwu

Python 3.5.2

"""

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import *
from mask import *
import argparse
from datetime import datetime
import logging
from timm.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
names = 'spiking_model'
data_path = '/home/hanbing/project/BP-for-SpikingNN-master2/raw/'  # todo: input your data path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0) 

_logger = logging.getLogger('train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"),'tu-mnist','conv-0.85-fc-1.9n'])
output_dir = get_outdir('./', 'train', exp_name)
setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))

best_acc = 0  # best test accuracy
best=0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
acc_record1 = list([])
loss_train_record = list([])
loss_test_record = list([])
pruning=0.1
rate_decay = 600
epoch_prune = 1
NUM=0

snn = SCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

m = Mask(snn)
m.init_length()

th1 = torch.zeros(batch_size,cfg_fc[0], device=device)
wwfc1 = torch.zeros_like(snn.fc1.weight, device=device)
thc1 = torch.zeros(batch_size,cfg_cnn[0][1], device=device)
wwconv1 = torch.zeros(cfg_cnn[0][1],cfg_cnn[0][0], device=device)
thc2 = torch.zeros(batch_size,cfg_cnn[1][1], device=device)
wwconv2 = torch.zeros(cfg_cnn[1][1],cfg_cnn[0][1], device=device)

c1_trace = torch.zeros(cfg_cnn[0][1], device=device)
c2_trace = torch.zeros(cfg_cnn[1][1], device=device)
h1_trace=torch.zeros(cfg_fc[0], device=device)

mask_index = [x for x in range(0, 6, 2)]
mat={}
for index, item in enumerate(snn.parameters()):
    if index in mask_index:
        mat[index]=torch.ones_like(item)
sumww=torch.zeros(588000, device=device)
sumnn=torch.zeros(300, device=device)
for epoch in range(num_epochs):
        
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        outputs, xmean, h1mean, c1_c, c2_c,cx_c= snn(images,mat, epoch=epoch, train=1)

        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        NUM=NUM+1
        post1 = (h1mean * (h1mean - th1))
        hebb1 = torch.mm(post1.T, xmean)
        wwfc1 =  wwfc1 + hebb1
        th1 = torch.div(th1 * (NUM - 1) + h1mean,NUM)

        cx=torch.sum(torch.sum(cx_c,dim=2),dim=2)
        c1=torch.sum(torch.sum(c1_c,dim=2),dim=2)
        c2=torch.sum(torch.sum(c2_c,dim=2),dim=2)

        postc1= (c1 * (c1 - thc1))
        hebbc1 = torch.mm(postc1.T, cx)
        wwconv1 = wwconv1 + hebbc1
        thc1 = torch.div(thc1 * (NUM - 1) + c1,NUM)

        postc2= (c2 * (c2 - thc2))
        hebbc2 = torch.mm(postc2.T, c1)
        wwconv2 =wwconv2 + hebbc2
        thc2 = torch.div(thc2 * (NUM - 1) + c2,NUM)

        h1_s = torch.sum(h1mean, dim=0)/batch_size
        c1_s = torch.sum(c1, dim=0)/batch_size
        c2_s = torch.sum(c2, dim=0)/batch_size

        h1_trace =  h1_trace + h1_s
        c1_trace =c1_trace + c1_s
        c2_trace =c2_trace + c2_s
        
        if (i+1)%rate_decay==0 and epoch>0:
            wfc1 = torch.sum(wwfc1, dim=1)
            wfc1=unit_tensor(wfc1)
            h1_trace=unit_tensor(h1_trace)
            wfc11=wfc1*h1_trace
    
            wconv1 = torch.sum(wwconv1, dim=1)
            wconv1=unit_tensor(wconv1)
            c1_trace=unit_tensor(c1_trace)
            wconv11=wconv1*c1_trace

            wconv2 = torch.sum(wwconv2, dim=1)
            wconv2=unit_tensor(wconv2)
            c2_trace=unit_tensor(c2_trace)
            wconv22=wconv2*c2_trace

            m.model = snn
            sumww=sumww+wwfc1.view(-1)
            sumnn=sumnn+wfc11
            matt,bcmww,bcmnn=m.init_mask(pruning,wfc11,wwconv2,wconv11,wconv22,wwfc1,epoch)
            m.do_mask() 
            snn = m.model
            # print(bcmww)
            # state = {
            #     'bcmw': bcmww,
            #     'bcmn': bcmnn,
            # }
            # torch.save(state, './checkpoint/ckpt' + 'ww' + '.t7')   

            for index, item in enumerate(snn.parameters()):
                if (index in mask_index):
                    mat[index]=matt[index]
            
            wwfc1 = torch.zeros_like(snn.fc1.weight, device=device)
            wwconv1 = torch.zeros(cfg_cnn[0][1],cfg_cnn[0][0], device=device)
            wwconv2 = torch.zeros(cfg_cnn[1][1],cfg_cnn[0][1], device=device)
            c1_trace = torch.zeros(cfg_cnn[0][1], device=device)
            c2_trace = torch.zeros(cfg_cnn[1][1], device=device)
            h1_trace=torch.zeros(cfg_fc[0], device=device)
        

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss))
            running_loss = 0
            print('Time elasped:', time.time() - start_time)
            '''m.model = snn
            #m.if_zero()
            m.init_mask(comp_rate_conv, comp_rate_full, wfc1, c1sum, c2sum)
            m.do_mask()
            #m.if_zero()
            snn = m.model
            if device == 'cuda':
                snn = snn.cuda()'''
        
    correct = 0
    total = 0
    cc=m.if_zero()
    if epoch %30 == 0 and epoch >1:
        pruning = pruning - 0.05
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs,mat)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx % 100 == 0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

    print('Iters:', epoch, '\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if best_acc < acc:
        best_acc = acc

    _logger.info('*** epoch: {0} (pruning {1},acc:{2})'.format(epoch, cc,acc))
    if epoch % 5 == 0:
        print(best_acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
    
        

