import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''convlayer = [0, 2, 4, 6]
fclayer=[7,8,9]
imgsize = [64, 32, 16,8]
batch = 20
step = 16
delta = 0.5
size = [2,128, 256, 256, 512]
fcsize=[512*8*8,512,100]'''
convlayer = [0, 1, 3, 4, 6, 7]
fclayer=[8,9]
imgsize = [32, 32, 16, 16, 8, 8]
batch = 50
step = 8
delta = 0.5
size = [3,128, 128, 256, 256, 512,512]
fcsize=[512*8*8,512]

def unit(x):
    maxx=torch.max(x)
    minx=torch.min(x)
    marge=maxx-minx
    xx=(x-minx)/marge
    return xx

class Mask:
    def __init__(self, model):
        self.model = model
        self.count_thre=20
        self.fullbook={}
        self.mat = {}
        self.feature=model.feature
        self.filter_indexs_his={}
        self.filter_indww_his={}
        self.n_index = {}
        self.ww_index={}
        self.fc={}
        self.fc[1]=model.fc2[0]
        self.fc[2]=model.fc[0]

    def init_length(self):
        for i in range(len(convlayer)):
            index=convlayer[i]
            self.fullbook[index] = np.ones((size[i+1],size[i],3,3))
            self.n_index[index] = np.zeros(size[i+1])
            self.filter_indexs_his[index] = np.array([size[i+1] + 1])
        for i in range(1,len(fclayer)):
            ind=fclayer[i]
            self.fullbook[ind] = np.ones((fcsize[i],fcsize[i-1]))
            self.n_index[ind] = np.zeros(fcsize[i])
            self.ww_index[ind]=np.zeros(fcsize[i]*fcsize[i-1])
            self.filter_indexs_his[ind] = np.array([fcsize[i] + 1])
            self.filter_indww_his[ind] = np.array([fcsize[i]*fcsize[i-1] + 1])
            
            
        

    def get_filter_codebook(self,ww,comp_rate, convtra,ii,index, wwf): 
        if len(ww.size()) == 4:
            filter_pruned_num = int(ww.size()[0] * comp_rate[ii])
            filters = torch.zeros(ww.size()[0]).cuda()
            www= torch.zeros(ww.size()[0]).cuda()
            weight_torch=abs(ww)
            wsum=torch.sum(weight_torch,dim=1)
            wsum=torch.sum(wsum,dim=1)
            wsum=torch.sum(wsum,dim=1)
            wsum=unit(wsum)
            convtra[ii]=unit(convtra[ii])
            for i in range(0, ww.size()[0]):
                if wsum[i] == 0:
                    filters[i] = 0
                else:
                    filters[i] = convtra[ii][i].cuda()*wsum[i].detach()
                    www[i]=wsum[i].detach()
            filters = filters.detach()
            filters = filters.cpu().numpy()
            filter_indexs = filters.argsort()[:filter_pruned_num]
            
        if len(ww.size()) == 2:
            filter_pruned_num = int(ww.size()[0] * comp_rate[ii])
            filter_indexs = convtra[index].argsort()[:filter_pruned_num].cpu().numpy()
            
            length=wwf.size()[0]*wwf.size()[1]
            filter_pruned = int(length * (1 - comp_rate[ii]))
            filter_ww= wwf.view(length).cpu().numpy()
            filter_indww = filter_ww.argsort()[:filter_pruned]
            book=np.ones(length)
                
            pos_indww = set(filter_indww) & set(self.filter_indww_his[index])  # calculate intersection  consectively
            pos_zeroww = set([i for i in range(length)]) - pos_indww
            pos_indww = np.array(list(pos_indww))
            pos_zeroww = np.array(list(pos_zeroww))
            if pos_zeroww.size > 0:
                self.ww_index[index][pos_zeroww] = 0
            if pos_indww.size > 0:
                self.ww_index[index][pos_indww] = self.ww_index[index][pos_indww] + 1  # intewrsection +1
            filter_indww = np.where(self.ww_index[index] >= self.count_thre)  # count>threshold
            book[filter_indww]=0
            book=book.reshape((ww.size()[0],-1))
            print(book.shape)
            self.fullbook[index]=self.fullbook[index]*book
            

        pos_index = set(filter_indexs) & set(self.filter_indexs_his[index])  # calculate intersection  consectively
        pos_zero = set([i for i in range(ww.size()[0])]) - pos_index
        pos_index = np.array(list(pos_index))
        pos_zero = np.array(list(pos_zero))
        if pos_zero.size > 0:
            self.n_index[index][pos_zero] = 0
        if pos_index.size > 0:
            self.n_index[index][pos_index] = self.n_index[index][pos_index] + 1  # intewrsection +1
        filter_ind = np.where(self.n_index[index] >= self.count_thre)  # count>threshold

        for x in range(0, len(filter_ind[0])):
            self.fullbook[index][filter_ind[0][x]] = 0
        self.n_index[index][filter_ind] = 0  # intrsection count set to 0
        self.filter_indexs_his[index] = filter_indexs  # update previous


        return self.fullbook[index]

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_mask(self, comp_rate_conv,comp_rate_fc,wwfc,convtra):
        for i in range(len(convlayer)):
            index=convlayer[i]
            ww = self.feature[index].conv.weight
            self.mat[index]=self.get_filter_codebook(ww, comp_rate_conv,convtra,i,index, wwf=0)
            self.mat[index] = self.convert2tensor(self.mat[index]).cuda()
        for i in range(1,len(fclayer)):
            ind=fclayer[i]
            ww = self.fc[i].fc.weight
            wwf=wwfc[i]
            self.mat[ind]=self.get_filter_codebook(ww,comp_rate_fc,convtra,i,ind, wwf)
            self.mat[ind] = self.convert2tensor(self.mat[ind]).cuda()

    def do_mask(self):
        for i in range(len(convlayer)):
            index=convlayer[i]
            ww = self.feature[index].conv.weight
            maskww=ww*self.mat[index]
            self.feature[index].conv.weight.data=maskww
        for i in range(1,len(fclayer)):
            ind=fclayer[i]
            ww = self.fc[i].fc.weight
            maskww=ww*self.mat[ind]
            self.fc[i].fc.weight.data=maskww

    def if_zero(self):
        for i in range(len(convlayer)):
            ww=self.feature[convlayer[i]].conv.weight
            a = ww.data.view(-1)
            b = a.cpu().numpy()
            print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
        
        for i in range(1,len(fcsize)):
            ww=self.fc[i].fc.weight
            a = ww.data.view(-1)
            b = a.cpu().numpy()
            print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
                
                
'''
ww=self.fc[i].fc.weight
num=0
for i in range(ww.size()[0]):
    w=torch.sum(ww[i])
    if w==0:
        num=num+1
    print(
        "zero weight is %d,number of weight is %d" % (num, ww.size()[0]))'''

