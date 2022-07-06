import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import sys
sys.path.append('..')

from SpikingNN.base.nodes import *
from SpikingNN.common import *
from SpikingNN.base.criterions import *
from SpikingNN.datasets.datasets import *
from SpikingNN.model_zoo.resnet import *
from SpikingNN.model_zoo.convnet import *

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"),'dvs-c10','conv-0.65-12-fc-1.5-13'])
output_dir = get_outdir('./', 'train', exp_name)
log = open(os.path.join(output_dir, '.txt'), 'w')

def print_log(print_string, log):
    #print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

# from ptflops import get_model_complexity_info
from thop import profile, clever_format
print("sss")
model='cifar_convnet'
dataset='cifar10'
num_classes=10
step=8
encode='direct'
node_type='PLIFNode'
thresh=0.5
tau=2.0
torch.backends.cudnn.benchmark = True
devicee=0
seed=42
torch.cuda.set_device('cuda:%d' % devicee)
torch.manual_seed(seed)

model = create_model(
    model,
    pretrained=False,
    num_classes=num_classes,
    dataset=dataset,
    step=step,
    encode_type=encode,
    node_type=eval(node_type),
    threshold=thresh,
    tau=tau,
    spike_output=True
)
model = model.cuda()
print(model)
channels = 2
lr=5e-3
batch_size=50
epochs=300
linear_scaled_lr = lr * batch_size/ 1024.0
config_parser = cfg = argparse.ArgumentParser(description='Training Config', add_help=False)
cfg.opt='adamw'
cfg.lr=linear_scaled_lr
cfg.weight_decay=0.01
cfg.momentum=0.9
cfg.epochs=epochs
cfg.sched='cosine'
cfg.min_lr=1e-5
cfg.warmup_lr=1e-6
cfg.warmup_epochs=5
cfg.cooldown_epochs=10
cfg.decay_rate=0.1
optimizer = create_optimizer(cfg, model)
lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)

loader_train, loader_eval = eval('get_%s_data' % dataset)(batch_size=batch_size, step=step)
print(len(loader_train), len(loader_eval))
train_loss_fn = UnilateralMse(1.)
validate_loss_fn = UnilateralMse(1.)

eval_metric='top1'
best_test = 0
best_testepoch = 0
best_testprun = 0
best_testepochprun = 0

from maskup import *
from traceup import *
m = Mask(model)
m.init_length()
tra=trace(model)
epoch_prune = 1

NUM = 0
rate_decay_epoch=30
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

    
def train_epoch(
        epoch, model, loader, optimizer, loss_fn,tra,NUM,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
        output,spikes = model(inputs)

        #tra.init()
        NUM = NUM + 1
        csum,fcsum= tra.computing_trace(spikes)
        for i in range(1,len(convlayer)):
            index=convlayer[i]
            #print(index,csum[index].size(),csum[index-1].size())
            post1 = (csum[index] * (csum[index] - th[index]))
            hebb1 = torch.mm(post1.T, csum[index-1]) #csum[convlayer[i - 1]]
            wwfc[index] = wwfc[index] + hebb1
            th[index] = torch.div(th[index] * (NUM - 1) + csum[index], NUM)
            cs=torch.sum(csum[index],dim=0)
            epoch_tra[index] = epoch_tra[index] + cs

        for i in range(1,len(fclayer)):
            index = fclayer[i]
            post1 = (fcsum[index] * (fcsum[index] - th[index]))
            hebb1 = torch.mm(post1.T, fcsum[fclayer[i - 1]])
            wwfc[index] = wwfc[index] + hebb1
            th[index] = torch.div(th[index] * (NUM - 1) + fcsum[index], NUM)
            cs=torch.sum(fcsum[index],dim=0)
            epoch_tra[index] = epoch_tra[index] + cs

        loss = loss_fn(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_m.update(loss.item(), inputs.size(0))
        top1_m.update(acc1.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx %40 == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print(epoch,loss.data,losses_m.avg,acc1, top1_m.avg,lr)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)]),epoch_tra,wwfc,NUM

def validate(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            #print(inputs.size())
            # inputs = inputs.type(torch.float64)
            last_batch = batch_idx == last_idx
            inputs = inputs.type(torch.FloatTensor).cuda()
            target = target.cuda()

            output,spikes = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            if last_batch or batch_idx %100 == 0:
                print(loss.data,losses_m.avg,acc1, top1_m.avg)

            batch_time_m.update(time.time() - end)
            end = time.time()


    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


for epoch in range(epochs):

    train_metrics, epoch_tra, wwfc, N = train_epoch(
        epoch, model, loader_train, optimizer, train_loss_fn,tra,NUM,
        lr_scheduler=lr_scheduler)
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
        m.model = model
        m.init_mask(wwfc,convtra,epoch)
        m.do_mask()
        cc=m.if_zero()
        model = m.model

    eval_metrics = validate(model, loader_eval, validate_loss_fn)
    top1=eval_metrics['top1']
    if top1 > best_testprun:
        best_testprun = top1
        best_testepochprun =epoch
    if epoch%40==0:
        print('pruning:',best_testprun,best_testepochprun)
    if epoch>4:
        print_log('*** epoch: {0} (pruning {1},acc:{2})'.format(epoch, cc,top1), log)
    if lr_scheduler is not None:
        lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
