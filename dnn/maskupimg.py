import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
convlayer = [0, 1, 3, 4, 6, 7]
fclayer=12
step = 8
delta = 0.5

class Mask:
    def __init__(self, model):
        self.model = model
        self.count_thre=15
        self.fullbook={}
        self.mat = {}
        self.feature=model.feature
        self.fc=model.fc
        self.filter_indexs_his={}
        self.n_index = {}
        self.size={}

    def init_length(self):
        for index in convlayer:
            ww=self.feature[index].conv.weight
            self.size[index]=ww.size()[0]
            self.fullbook[index] =torch.ones_like(ww,device=device)
            self.n_index[index] = torch.zeros(self.size[index],device=device)
            self.filter_indexs_his[index] = torch.tensor([self.size[index] + 1],device=device)
        ww = self.fc.fc.weight
        fcsize=ww.size()[0]
        self.fullbook[fclayer]=torch.ones_like(ww,device=device)
        self.n_index[fclayer] = torch.zeros(fcsize,device=device)
        self.filter_indexs_his[fclayer] = torch.tensor([fcsize + 1],device=device)
        return self.size,fcsize

    def get_filter_codebook(self,ww, comp_rate, convtra,index):
        filter_pruned_num = int(ww.size()[0] * (1 - comp_rate))
        if len(ww.size()) == 4:
            filters = torch.zeros(ww.size()[0], device=device)
            for i in range(0, ww.size()[0]):
                wsum = torch.sum(abs(ww[i]))
                if wsum == 0:
                    filters[i] = 0
                else:
                    #print(i,convtra[index][i])
                    filters[i] = convtra[index][i] * wsum.detach()
            filters = filters.detach()
            filter_indexs = filters.argsort()[:filter_pruned_num]

        if len(ww.size()) == 2:
            filter_indexs = (convtra.argsort()[:filter_pruned_num])

        pos_index = set(filter_indexs) & set(self.filter_indexs_his[index])  # calculate intersection  consectively
        pos_zero = set([i for i in range(ww.size()[0])]) - pos_index
        pos_index = torch.tensor(list(pos_index))
        pos_zero =torch.tensor(list(pos_zero))
        if pos_zero.size()[0] > 0:
            self.n_index[index][pos_zero] = 0
        if pos_index.size()[0] > 0:
            self.n_index[index][pos_index] = self.n_index[index][pos_index] + 1  # intewrsection +1
        filter_ind = torch.nonzero(self.n_index[index] >= self.count_thre)  # count>threshold

        for x in range(0, filter_ind.size()):
            self.fullbook[index][filter_ind[x]] = 0
        self.n_index[index][filter_ind] = 0  # intrsection count set to 0
        self.filter_indexs_his[index] = filter_indexs  # update previous

        return self.fullbook[index]


    def init_mask(self, comp_rate,convtra,wfc1):
        for index in convlayer:
            ww = self.feature[index].conv.weight
            self.mat[index]=self.get_filter_codebook(ww, comp_rate,convtra,index)
        ww = self.fc.fc.weight
        self.mat[fclayer] = self.get_filter_codebook(ww, comp_rate, wfc1,index=fclayer)



    def do_mask(self):
        for index in convlayer:
            ww = self.feature[index].conv.weight
            maskww=ww*self.mat[index]
            self.feature[index].conv.weight.data=maskww
        index=fclayer
        ww = self.fc.fc.weight
        maskww = ww * self.mat[index]
        self.fc.fc.weight.data = maskww

    def if_zero(self):
        for index in convlayer:
            ww=self.feature[index].conv.weight
            ww=ww.view(ww.size()[0],-1)
            w=torch.sum(ww,dim=1)
            num=0
            for j in range(w.size()[0]):
                if w[j]==0:
                    num=num+1
            print("%d,zero weight is %d,number of weight is %d" % (index,num, w.size()[0]))
        ww = self.fc.fc.weight
        ww = ww.view(ww.size()[0], -1)
        w = torch.sum(ww, dim=1)
        num = 0
        for j in range(w.size()[0]):
            if w[j] == 0:
                num = num + 1
        print("%d,zero weight is %d,number of weight is %d" % (fclayer,num, w.size()[0]))



