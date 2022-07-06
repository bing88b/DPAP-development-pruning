import numpy as np
import torch
import math
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        maxx=np.percentile(x, 80)
        minx=np.percentile(x, 5)
        # maxx=np.max(x)
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
                filter_wwcon = wconv2.view(-1).cpu().numpy()
                delta_wwcon=(unit(filter_wwcon)*2-1.4)
                pos=np.where(delta_wwcon>0)[0]
                delta_wwcon[pos]=delta_wwcon[pos]+5
                self.reducewwcon= self.reducewwcon*0.999+delta_wwcon*math.exp(-int(epoch/10))
                print(self.reducewwcon.min())
                filter_indwwcon = np.where(self.reducewwcon < 0)[0]
                print(filter_indwwcon)
                for x in range(0, len(filter_indwwcon)):
                    self.fullbook[index][filter_indwwcon[x] * 9: (filter_indwwcon[x] + 1) * 9] = 0

                # wconv= wconv22.cpu().numpy()
                # print(wconv.mean(),wconv.max(), wconv.min())
                # self.n_delta[index]=(unit(wconv)*2- 1)
                # pos=np.where(self.n_delta[index]>0)[0]
                # self.n_delta[index][pos]=self.n_delta[index][pos]+5
                # self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int(epoch/10))
                # filter_ind = np.where(self.reduce[index] <0)[0]
                # print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),math.exp(-int(epoch/10)))
                # if len(filter_ind)!=0:
                #     print(len(filter_ind))
                #     print(wconv[filter_ind].mean(),np.max(wconv[filter_ind]),np.min(wconv[filter_ind]))
                # kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
                
                # for x in range(0, len(filter_ind)):
                #     self.fullbook[index][filter_ind[x] * kernel_length: (filter_ind[x] + 1) * kernel_length] = 0


        if len(weight_torch.size()) == 2:
            if index == 4:
                filter_ww = wwfc1.view(length).cpu().numpy()
                print(filter_ww.mean(),filter_ww.max(),filter_ww.min())
                delta_ww=(unit(filter_ww)*2-0.4)
                print(delta_ww.mean(),delta_ww.max(),delta_ww.min())
                pos=np.where(delta_ww>0)[0]
                delta_ww[pos]=delta_ww[pos]+2
                self.reduceww= self.reduceww*0.999+delta_ww*math.exp(-int(epoch/10))
                dq.append(self.reduceww)
                filter_indww = np.where(self.reduceww < 0)[0]
                self.fullbook[index][filter_indww] = 0
                print(self.reduceww.mean(),self.reduceww.max(),self.reduceww.min(),pruning,math.exp(-int(epoch/10)))
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
                
                # wfc1= wfc1.cpu().numpy()
                # #print( wfc1.mean(), wfc1.max(), wfc1.min())
                # self.n_delta[index]=(unit(wfc1)*2-0.915)
                # pos=np.where(self.n_delta[index]>0)[0]
                # self.n_delta[index][pos]=self.n_delta[index][pos]+2
                # self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int(epoch/10))
                # dqn.append(self.reduce[index])
                # filter_ind = np.where(self.reduce[index] < 0)[0]
                # kernel_length = weight_torch.size()[1]

                # nn=set(filter_ind)-set(self.his_nn)
                # bcmn=[]
                # for xx in dqn.data():
                #     bcmn.append(xx[list(nn)])
                # for i in range(len(bcmn[0])):
                #     c=[]
                #     for j in range(8):
                #         c.append(bcmn[j][i])
                #     self.bcmnn.append(c)
                # d=np.array(self.bcmnn)
                # print(d.shape)
                # print(np.mean(d,axis=0))
                # self.filter_ind =filter_ind
                # self.his_nn=filter_ind
                
                # for x in range(0, len(filter_ind)):
                #     self.fullbook[index][filter_ind[x] * kernel_length: (filter_ind[x] + 1) * kernel_length] = 0
                
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
                    self.reduce[index1] =5*np.ones(self.model_size[index1][0])
                    self.filter_indexs_his[index1] = np.array([self.model_size[index1][0] + 1])
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
            self.fullbook[index1] = np.ones(self.model_length[index1])

        self.reduceww=5*np.ones(self.model_length[4])
        self.reducewwcon=5*np.ones(int(self.model_length[2]/9))
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

    def init_mask(self, pruning, wfc1=0,wconv2=0, wconv11=0,wconv22=0,wwfc1=0,epoch=0):
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



# import numpy as np
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Mask:
#     def __init__(self, model):
#         self.model_size = {}
#         self.model_length = {}
#         self.compress_rate = {}
#         self.mat = {}
#         self.model = model
#         self.mask_index = []
#         self.n_index={}
#         self.count_thre=15
#         self.fullbook={}
#         self.filter_indexs_his={}
        

#     def get_codebook(self, weight_torch, compress_rate, length):
#         weight_vec = weight_torch.view(length)
#         weight_np = weight_vec.cpu().numpy()

#         weight_abs = np.abs(weight_np)
#         weight_sort = np.sort(weight_abs)

#         threshold = weight_sort[int(length * (1 - compress_rate))]
#         weight_np[weight_np <= -threshold] = 1
#         weight_np[weight_np >= threshold] = 1
#         weight_np[weight_np != 1] = 0


#         return weight_np

#     def get_filter_codebook(self, weight_torch, comp_rate, length,wfc1,c1_trace=0,c2_trace=0,wwfc1=0,index=0):
#         codebook = np.ones(length)
#         if len(weight_torch.size()) == 4:
#             filter_pruned_num = int(weight_torch.size()[0] * (1 - comp_rate))
#             filters = torch.zeros(weight_torch.size()[0], device=device)
#             for i in range(0, weight_torch.size()[0]):
#                 wsum = torch.sum(abs(weight_torch[i]))
#                 if wsum==0:
#                     filters[i]=0
#                 else:
#                     if index == 0:
#                         filters[i] = c1_trace[i]*wsum
#                     if index == 2:
#                         filters[i] = c2_trace[i]*wsum
#             filters = filters.cpu().numpy()
#             filter_indexs = filters.argsort()[:filter_pruned_num]
            
#             pos_index = set(filter_indexs) & set(self.filter_indexs_his[index])  # calculate intersection  consectively
#             pos_zero = set([i for i in range(weight_torch.size()[0])]) - pos_index
#             pos_index = np.array(list(pos_index))
#             pos_zero = np.array(list(pos_zero))
#             if pos_zero.size > 0:
#                 self.n_index[index][pos_zero] = 0
#             if pos_index.size > 0:
#                 self.n_index[index][pos_index] = self.n_index[index][pos_index] + 1  # intewrsection +1
#             filter_ind = np.where(self.n_index[index] >= self.count_thre)  # count>threshold
#             kernel_length =weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]


#         if len(weight_torch.size()) == 2:
#             if index==4:
#                 filter_pruned_num = int(wfc1.size()[0] * (1 - comp_rate))
#                 filter_indexs = (wfc1.argsort()[:filter_pruned_num]).cpu().numpy()
                
#                 filter_pruned = int(length * (1 - comp_rate))
#                 filter_ww= wwfc1.view(length)
#                 ilter_indww = (filter_ww.argsort()[:filter_pruned]).cpu().numpy()
#                 pos_indww = set(ilter_indww) & set(self.filter_indww_his)  # calculate intersection  consectively
#                 pos_zeroww = set([i for i in range(length)]) - pos_indww
#                 pos_indww = np.array(list(pos_indww))
#                 pos_zeroww = np.array(list(pos_zeroww))
#                 if pos_zeroww.size > 0:
#                     self.ww_index[pos_zeroww] = 0
#                 if pos_indww.size > 0:
#                     self.ww_index[pos_indww] = self.ww_index[pos_indww] + 1  # intewrsection +1
#                 filter_indww = np.where(self.ww_index>= self.count_thre)  # count>threshold
#                 self.fullbook[index][filter_indww]=0
                

#             if index==6:
#                 #filter_pruned_num = int(wfc2.size()[0] * (1 - comp_rate))
#                 filter_indexs = []#wfc2.argsort()[:filter_pruned_num]
                
#             pos_index = set(filter_indexs) & set(self.filter_indexs_his[index])  # calculate intersection  consectively
#             pos_zero = set([i for i in range(weight_torch.size()[0])]) - pos_index
#             pos_index = np.array(list(pos_index))
#             pos_zero = np.array(list(pos_zero))
#             if pos_zero.size > 0:
#                 self.n_index[index][pos_zero] = 0
#             if pos_index.size > 0:
#                 self.n_index[index][pos_index] = self.n_index[index][pos_index] + 1  # intewrsection +1
#             filter_ind = np.where(self.n_index[index] >= self.count_thre)  # count>threshold
#             kernel_length = weight_torch.size()[1]
                
#         for x in range(0, len(filter_ind[0])):
#             self.fullbook[index][filter_ind[0][x] * kernel_length: (filter_ind[0][x] + 1) * kernel_length] = 0
#         self.n_index[index][filter_ind] = 0  # intrsection count set to 0
#         self.filter_indexs_his[index] =filter_indexs  # update previous

#         return self.fullbook[index]



#     def convert2tensor(self, x):
#         x = torch.FloatTensor(x)
#         return x

#     def init_length(self):
#         for index, item in enumerate(self.model.parameters()):
#             self.model_size[index] = item.size()

#         for index1 in self.model_size:
#             for index2 in range(0, len(self.model_size[index1])):
#                 if index2 == 0:
#                     self.model_length[index1] = self.model_size[index1][0]
#                     self.n_index[index1]=np.zeros(self.model_size[index1][0])
#                     self.filter_indexs_his[index1] = np.array([self.model_size[index1][0]+1])
#                 else:
#                     self.model_length[index1] *= self.model_size[index1][index2]
#             self.fullbook[index1]=np.ones(self.model_length[index1])
#         self.filter_indww_his= np.array([self.model_length[4]+1])
#         self.ww_index=np.zeros(self.model_length[4])

#     def init_rate(self, comp_rate_conv,comp_rate_full):
#         for index, item in enumerate(self.model.parameters()):
#             self.compress_rate[index] = 1
#         self.compress_rate[0] = comp_rate_conv
#         self.compress_rate[2] = comp_rate_conv
#         self.compress_rate[4] = comp_rate_full
#         self.compress_rate[6] = comp_rate_full
#         self.mask_index = [x for x in range(0, 8, 2)]

#     #        self.mask_index =  [x for x in range (0,330,3)]

#     def init_mask(self, comp_rate_conv,comp_rate_full,wfc1=0,c1_trace=0,c2_trace=0,wwfc1=0):
#         self.init_rate(comp_rate_conv,comp_rate_full)
#         for index, item in enumerate(self.model.parameters()):
#             if (index in self.mask_index):
#                 self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
#                                                            self.model_length[index],wfc1,c1_trace,c2_trace,wwfc1,index)
#                 self.mat[index] = self.convert2tensor(self.mat[index])
#                 #print("filter codebook done")
#                 if device=='cuda':
#                     self.mat[index] = self.mat[index].cuda()
#         #print("mask Ready")

#     def do_mask(self):
#         for index, item in enumerate(self.model.parameters()):
#             if (index in self.mask_index):
#                 #if index==0:
#                     #print(item.data)
#                 a = item.data.view(self.model_length[index])
#                 b = a * self.mat[index].cuda()
#                 item.data = b.view(self.model_size[index])
#                 #if index==0:
#                     #print(item.data)
#         #print("mask Done")

#     def if_zero(self):
#         for index, item in enumerate(self.model.parameters()):
#             #            if(index in self.mask_index):
#             if len(item.size())>1:
#                 a = item.data.view(self.model_length[index])
#                 b = a.cpu().numpy()
#                 print(
#                     "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
