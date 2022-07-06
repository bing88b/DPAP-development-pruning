import numpy as np
import torch
import math
from collections import deque

# 先进先出队列
class FifoQue(object):
    def __init__(self, max_size=10):
        self.dq = deque()
        self.max_size = max_size

    def append(self, item):
        if len(self.dq) >= self.max_size:
            self.dq.popleft()
        self.dq.append(item)

    def data(self):
        return self.dq

dq = FifoQue(max_size=8)
dqn = FifoQue(max_size=8)

def unit(x):
    if x.size>0:
        maxx=np.percentile(x, 99.9)
        minx=np.percentile(x, 0.1)
        # minx=np.min(x)
        marge=maxx-minx
        if marge!=0:
            xx=(x-minx)/marge
            xx=np.clip(xx, 0,1)
        else:
            xx=0.5*np.ones_like(x)
        return xx
    else:
        return x
        
def unit_tensor(x):
    if x.size()[0]>0:
        maxx=torch.max(x)
        minx=torch.min(x)
        marge=maxx-minx
        if marge!=0:
            xx=(x-minx)/marge
        else:
            xx=0.5*torch.ones_like(x)
        return xx
    else:
        return x

class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.n_index = {}
        self.fullbook = {}
        self.filter_indexs_his = {}
        self.filter_his={}
        self.n_delta={}
        self.reduce={}
        self.maskk={}
        self.bcmww=[]
        self.bcmnn=[]
        self.alll=[i for i in range(300)]

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        return weight_np

    def get_filter_codebook(self, weight_torch, pruning, length, wfc1, wconv2,wconv11=0,wconv22=0, wwfc1=0, index=0,epoch=0):
        if len(weight_torch.size()) == 4:
            # if index==0:
            #     wconv= wconv11.cpu().numpy()
            if index==2:
                # filter_wwcon = wconv2.view(-1).cpu().numpy()
                # delta_wwcon=(unit(filter_wwcon)*2-0.85)
                # pos=np.where(delta_wwcon>0)[0]
                # delta_wwcon[pos]=delta_wwcon[pos]+5
                # self.reducewwcon= self.reducewwcon*0.999+delta_wwcon*math.exp(-int(epoch/10))
                # print(self.reducewwcon.min())
                # filter_indwwcon = np.where(self.reducewwcon < 0)[0]
                # print(filter_indwwcon)
                # for x in range(0, len(filter_indwwcon)):
                #     self.fullbook[index][filter_indwwcon[x] * 9: (filter_indwwcon[x] + 1) * 9] = 0

                wconv=wconv22.cpu().numpy()
                print(wconv.shape)
                self.n_delta[index]=(unit(wconv)*2- 0.35)
                pos=np.where(self.n_delta[index]>0)[0]
                self.n_delta[index][pos]=self.n_delta[index][pos]+5
                self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch)/10))
                filter_ind = np.where(self.reduce[index] <0)[0]
                print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min())
                if len(filter_ind)!=0:
                    print(filter_ind)
                    #print(wconv[filter_ind].mean(),np.max(wconv[filter_ind]),np.min(wconv[filter_ind]))
                kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
                
                for x in range(0, len(filter_ind)):
                    self.fullbook[index][filter_ind[x] * kernel_length: (filter_ind[x] + 1) * kernel_length] = 0


        if len(weight_torch.size()) == 2:
            if index == 4:
                filter_ww = wwfc1.view(length).cpu().numpy()
                print(filter_ww.mean(),filter_ww.max(),filter_ww.min())
                delta_ww=(unit(filter_ww)*2-0.625)
                pos=np.where(delta_ww>0)[0]
                delta_ww[pos]=delta_ww[pos]+2
                self.reduceww= self.reduceww*0.999+delta_ww*math.exp(-int((epoch)/10))
                print(self.reduceww.mean(),self.reduceww.max(),self.reduceww.min(),pruning,math.exp(-int(epoch/10)))
                dq.append(self.reduceww)
                filter_indww = np.where(self.reduceww < 0)[0]
                self.fullbook[index][filter_indww] = 0
                # ww=set(filter_indww)-set(self.his_ww)
                # bcmw=[]
                # for xx in dq.data():
                #     bcmw.append(xx[list(ww)])
                # for i in range(len(bcmw[0])):
                #     a=[]
                #     for j in range(8):
                #         a.append(bcmw[j][i])
                #     self.bcmww.append(a)
                # b=np.array(self.bcmww)
                # print(b.shape)
                # print(np.mean(b,axis=0))
                # self.his_ww=filter_indww
                
                wfc1= wfc1.cpu().numpy()
                self.n_delta[index]=(unit(wfc1)*2-0.625)
                pos=np.where(self.n_delta[index]>0)[0]
                self.n_delta[index][pos]=self.n_delta[index][pos]+2
                self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch)/10))
                print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),math.exp(-int(epoch/10)))
                # dqn.append(self.reduce[index])
                filter_ind = np.where(self.reduce[index] < 0)[0]
                kernel_length = weight_torch.size()[1]

                nn=set(filter_ind)-set(self.his_nn)
                # bcmn=[]
                alnn=[]
                self.alll=set(self.alll)-nn
                for i in range(len(nn)):
                    noz=np.where(self.fullbook[index][list(nn)[i] * kernel_length: (list(nn)[i] + 1) * kernel_length]!=0)[0]
                    nol=len(noz)
                    self.bcmnn.append(nol)
                for j in range(len(self.alll)):
                    noza=np.where(self.fullbook[index][list(self.alll)[j] * kernel_length: (list(self.alll)[j] + 1) * kernel_length]!=0)[0]
                    nola=len(noza)
                    alnn.append(nola)
                # # for xx in dqn.data():
                # #     bcmn.append(xx[list(nn)])
                # for i in range(len(bcmn[0])):
                #     c=[]
                #     for j in range(8):
                #         c.append(bcmn[j][i])
                #     self.bcmnn.append(c)
                # d=np.array(self.bcmnn)
                print(len(self.bcmnn))
                print(np.mean(np.array(self.bcmnn),axis=0))
                print(len(alnn))
                print(np.mean(np.array(alnn),axis=0))
                self.filter_ind =filter_ind
                self.his_nn=filter_ind

                for x in range(0, len(filter_ind)):
                    self.fullbook[index][filter_ind[x] * kernel_length: (filter_ind[x] + 1) * kernel_length] = 0
                
        return self.fullbook[index],self.bcmww,self.bcmnn

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                    self.n_index[index1] = np.zeros(self.model_size[index1][0])
                    self.n_delta[index1] = np.zeros(self.model_size[index1][0])
                    self.filter_his[index1] = np.zeros(self.model_size[index1][0])
                    self.reduce[index1] = 5*np.ones(self.model_size[index1][0])
                    self.filter_indexs_his[index1] = np.array([self.model_size[index1][0] + 1])
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
            self.fullbook[index1] = np.ones(self.model_length[index1])

        self.reduceww=5*np.ones(self.model_length[4])
        self.tww=np.zeros((10,self.model_length[4]))
        self.reducewwcon=5*np.ones(int(self.model_length[2]/9))
        self.filter_ind=[]
        self.filter_indww=[]
        self.his_ww=[]
        self.his_nn=[]

    def init_rate(self, comp_rate_conv, comp_rate_full):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
        self.compress_rate[0] = comp_rate_conv
        self.compress_rate[2] = comp_rate_conv
        self.compress_rate[4] = comp_rate_full
        self.mask_index = [x for x in range(0, 6, 2)]
    #        self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, pruning, wfc1=0, wconv2=0,wconv11=0,wconv22=0,wwfc1=0,epoch=0):
        # print(wfc1)
        # print(wconv22)
        # print(wwfc1)
        #self.init_rate(comp_rate_conv, comp_rate_full)
        self.mask_index = [x for x in range(0, 6, 2)]
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.mat[index],bcmww,bcmnn = self.get_filter_codebook(item.data, pruning,
                                                           self.model_length[index], wfc1, wconv2,wconv11,wconv22,wwfc1,
                                                           index,epoch)
                self.mat[index] = self.convert2tensor(self.mat[index])
                # print("filter codebook done")
                self.mat[index] = self.mat[index].cuda()
                self.maskk[index]= self.mat[index].view(self.model_size[index])
        return self.maskk,bcmww,bcmnn
        # print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index==0:
                # print(item.data)
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index].cuda()
                item.data = b.view(self.model_size[index])
                # if index==0:
                # print(item.data)
        # print("mask Done")

    def if_zero(self):
        cc=[]
        for index, item in enumerate(self.model.parameters()):
            #            if(index in self.mask_index):
            if len(item.size()) > 1:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
                cc.append(len(b) - np.count_nonzero(b))
        return cc
                
