import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

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

from mask import *
from trace import *
m = Mask(model)
m.init_length()
tra=trace(model)
epoch_prune = 1
#comp_rate_conv = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#comp_rate_fc = [0.0, 0.0, 0.0]
comp_rate_conv = [0.1, 0.1, 0.1, 0.10, 0.1, 0.1, 0.1, 0.1]
comp_rate_fc = [0.5, 0.5, 0.5]
NUM = 0
rate_decay_epoch=30
th={}
for i in range(1,len(fclayer)):
    th[i]=torch.zeros((batch,fcsize[i]),device=device)
convtra = {}
wwfc={}
for i in range(len(convlayer)):
    convtra[i] = torch.rand(size[i+1])
for i in range(1,len(fclayer)):
    ind=fclayer[i]
    convtra[ind]=torch.rand(fcsize[i])
    wwfc[i]=torch.rand(fcsize[i],fcsize[i-1])
m.model = model
m.if_zero()
m.init_mask(comp_rate_conv,comp_rate_fc,wwfc,convtra)
m.do_mask()
m.if_zero()
model = m.model
    
def train_epoch(
        epoch, model, loader, optimizer, loss_fn,tra,NUM,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    epoch_tra = {}
    wwfc = {}
    for i in range(len(convlayer)):
        epoch_tra[i] = torch.zeros((size[i + 1], imgsize[i], imgsize[i])).cuda()
    for i in range(len(fclayer)):
        ind = fclayer[i]
        epoch_tra[ind] = torch.zeros(fcsize[i]).cuda()
        if i > 0:
            wwfc[i] = torch.zeros((fcsize[i], fcsize[i - 1])).cuda()

    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
        output,spikes = model(inputs)

        #tra.init()
        NUM = NUM + 1
        ctra,fcsum,fctra = tra.computing_trace(spikes)
        for i in range(len(convlayer)):
            epoch_tra[i] = epoch_tra[i] + ctra[i]
        for i in range(len(fclayer)):
            ind = fclayer[i]
            epoch_tra[ind] = epoch_tra[ind] + fcsum[i]
            if i > 0:
                post1 = (fctra[i] * (fctra[i] - th[i]))
                hebb1 = torch.mm(post1.T, fctra[i - 1])
                wwfc[i] = wwfc[i] + hebb1
                th[i] = torch.div(th[i] * (NUM - 1) + fctra[i], NUM)

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
    eval_metrics = validate(model, loader_eval, validate_loss_fn)
    top1=eval_metrics['top1']
    if top1 > best_testprun:
        best_testprun = top1
        best_testepochprun =epoch
    if epoch%40==0:
        print('pruning:',best_testprun,best_testepochprun)

    train_metrics, epoch_tra, wwfc, N = train_epoch(
        epoch, model, loader_train, optimizer, train_loss_fn,tra,NUM,
        lr_scheduler=lr_scheduler)

    eval_metrics = validate(model, loader_eval, validate_loss_fn)
    top1=eval_metrics['top1']
    if top1 > best_test:
        best_test = top1
        best_testepoch =epoch
    if epoch%40==0:
        print(best_test,best_testepoch)

    if lr_scheduler is not None:
        lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
    NUM = N
    convtra = {}
    for i in range(len(convlayer)):
        ccsum = torch.sum(epoch_tra[i], dim=1)
        convtra[i] = torch.sum(ccsum, dim=1)
    for i in range(1, len(fclayer)):
        ind = fclayer[i]
        wf = torch.sum(wwfc[i], dim=1)
        wf=unit(wf)
        epoch_tra[ind]=unit(epoch_tra[ind])
        convtra[ind] = wf * epoch_tra[ind]
    if (epoch % epoch_prune == 0):
        m.model = model
        m.if_zero()
        m.init_mask(comp_rate_conv, comp_rate_fc, wwfc,convtra)
        m.do_mask()
        m.if_zero()
        model = m.model
    if epoch % rate_decay_epoch == 0 and epoch > 1:
        for i in range(len(comp_rate_conv)):
            if (comp_rate_conv[i] - 0.1) > 0:
                comp_rate_conv[i] = comp_rate_conv[i] - 0.1
        for j in range(len(comp_rate_fc)):
            if (comp_rate_fc[j]-0.1)>0:
                comp_rate_fc[j] = comp_rate_fc[j] - 0.1
