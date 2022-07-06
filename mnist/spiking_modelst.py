import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.2  # decay constants
num_classes = 10
batch_size = 100
learning_rate = 1e-3
num_epochs = 150  # max epoch
delta = 0.5
deltas = 0.5
alpha = 0.001
pure_th = 0.9
decay_th = 0.5


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


# membrane potential update
def mem_update(ops, x, mem, spike):
    con=ops(x)
    mem = mem * decay * (1. - spike) + con
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike, con


# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 15, 1, 1, 3),
           (15, 40, 1, 1, 3), ]
# kernel size
cfg_kernel = [28, 14, 7]
# fc layer
cfg_fc = [300, 10]


# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def stdp(self,pres,posts,stdpwin=5,time_window=20):
        sizew=torch.mm(posts[0].T,pres[0])
        E=torch.zeros_like(sizew)
        tau_e = 5
        tau = 20  # time constant of STDP
        Apos = 0.925
        Aneg = 0.1
        sp=50
        for t in range(time_window):
            if t > stdpwin:
                #E = E - (E / tau_e)
                # ltp
                post_fire = posts[t]
                for iter in range(1, stdpwin + 1):
                    pre_fire = pres[t - iter]
                    s=torch.mm(post_fire.T,pre_fire)
                    # print(torch.max(s))
                    sw=torch.where(s>sp,1,0)
                    # s[post_fire][pre_fire]=s[post_fire][pre_fire]+Apos*np.exp(iter/tau)
                    E = E - E /tau_e
                    E = E+ Apos * np.exp(-iter /tau)*sw

                # ltd
                pre_fire = pres[t]
                for iter in range(1, stdpwin + 1):
                    post_fire = posts[t - iter]
                    s = torch.mm(post_fire.T, pre_fire)
                    sw = torch.where(s > sp, 1, 0)
                    # s[post_fire][pre_fire]=s[post_fire][pre_fire]-Aneg*np.exp(-iter/tau)
                    E = E - E / tau_e
                    E = E -Aneg  * np.exp(-iter / tau) * sw
        return E

    def forward(self, input, mat,time_window=20, epoch=0, train=0):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        cx_strace=torch.zeros(batch_size,cfg_cnn[0][0], cfg_kernel[0], cfg_kernel[0], device=device)
        c1_strace = torch.zeros(batch_size,cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_strace = torch.zeros(batch_size,cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        xsum=torch.zeros(batch_size, cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], device=device)
        h1sum=torch.zeros(batch_size, cfg_fc[0], device=device)

        h1summ=torch.zeros(time_window,batch_size, cfg_fc[0], device=device)
        xsumm=torch.zeros(time_window,batch_size, cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], device=device)
        c1summ =torch.zeros(time_window,batch_size, cfg_cnn[0][1], device=device)
        c2summ =torch.zeros(time_window,batch_size, cfg_cnn[1][1], device=device)

        for step in range(time_window):  # simulation time steps
            cx = input > torch.rand(input.size(), device=device)  # prob. firing
            
            self.conv1.weight.data = self.conv1.weight.data * mat[0]
            c1_mem, c1_spike,conc1 = mem_update(self.conv1, cx.float(), c1_mem, c1_spike)
            c1summ[step]=torch.sum(torch.sum(c1_spike,2),2)

            x = F.avg_pool2d(c1_spike, 2)
            
            self.conv2.weight.data = self.conv2.weight.data * mat[2]
            c2_mem, c2_spike,conc2 = mem_update(self.conv2, x, c2_mem, c2_spike)
            c2summ[step]=torch.sum(torch.sum(c2_spike,2),2)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)
            xsumm[step]=x
            
            self.fc1.weight.data = self.fc1.weight.data * mat[4]

            h1_mem, h1_spike,conh1 = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1summ[step]=h1_spike
            h2_mem, h2_spike,conh2 = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
            

            if (train != 0):
                cx_strace=delta*cx_strace+cx
                c1_strace=delta*c1_strace+c1_spike
                c2_strace=delta*c2_strace+c2_spike
                xsum=delta*xsum+x
                h1sum=delta*h1sum+ h1_spike


        outputs = h2_sumspike / time_window
        if (train != 0):
            cxsum=cx_strace/(time_window)
            c1sum=c1_strace/(time_window)
            c2sum=c2_strace/(time_window)

            xmean=xsum/(time_window)
            h1mean=h1sum/(time_window)

            c1sum = c1sum.detach()
            c2sum = c2sum.detach()
            h1mean = h1mean.detach()
            xmean = xmean.detach()

            wwc2=self.stdp(c1summ,c2summ)
            wwh1 = self.stdp(xsumm, h1summ)
            
            return outputs,xmean,h1mean,c1sum,c2sum,cxsum,wwc2,wwh1
        else:
            return outputs



