import numpy as np
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch = 100
delta = 0.5
layer=46

#convlayer = [-1,0, 2, 5, 7, 10, 12,14,17,19,21,24,26,28]
convlayer = [-1,0,3,6,9,13,16,19,22,26,29,32,35,38]
#convlayer = [-1,0,2,4,6, 9,11,13,16,18,20,23,25,27]
fclayer=[41,42,44]
fcc=[0,2]
#imgsize = [32,32, 32, 16, 16, 8, 8,8,4,4,4,2,2,2]
size = [3,64, 64, 128, 128, 256,256,256,512,512,512,512,512,512]
fcsize=[512*8*8,4096,4096]

def unit(x):
    if x.size()[0]>0:
        xnp=x.cpu().numpy()
        maxx=np.percentile(xnp, 75)
        minx=torch.min(x)
        marge=maxx-minx
        if marge!=0:
            xx=(x-minx)/marge
            xx=torch.clip(xx, 0,1)
        else:
            xx=0.5*torch.ones_like(x)
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
        self.model = model
        self.fullbook={}
        self.mat = {}
        self.feature=model.features
        self.fc=model.classifier
        self.n_delta={}
        self.ww_delta={}
        self.reduce={}
        self.reduceww={}

    def init_length(self):
        # self.mask_index =  [x for x in range (0,93,3)]
        # self.index=[]
        # for i in range(layer):
        #     if (i%3)!=0:
        #         self.index.append(i)
        #         self.fullbook[i] = np.ones((size[i],size[i-1],3,3))
        #         self.n_delta[i]=np.zeros(size[i])
        #         self.reduce[i] = 10*np.ones(size[i])
        for i in range(1,len(convlayer)):
            index=convlayer[i]
            self.fullbook[index] =torch.ones((size[i],size[i-1],3,3),device=device)
            self.n_delta[index]=torch.zeros(size[i],device=device)
            self.reduce[index] = 10*torch.ones(size[i],device=device)
        for i in range(1,len(fclayer)):
            index=fclayer[i]
            self.fullbook[index] = torch.ones((fcsize[i],fcsize[i-1]),device=device)
            self.n_delta[index]=torch.zeros(fcsize[i],device=device)
            self.ww_delta[index]=torch.zeros(fcsize[i]*fcsize[i-1],device=device)
            self.reduce[index] = 10*torch.ones(fcsize[i],device=device)
            self.reduceww[index] = 10*torch.ones(fcsize[i]*fcsize[i-1],device=device)

    def get_filter_codebook(self,ww,dendrite,ii,index,epoch): 
        if ii == 4:
            wconv= dendrite#.cpu().numpy()
            self.n_delta[index]=(unit(wconv)*2-0.6)
            pos=torch.nonzero(self.n_delta[index]>0)
            self.n_delta[index][pos]=self.n_delta[index][pos]+5
            print(wconv.mean(),wconv.max(), wconv.min())
            self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch-5)/13))
            filter_ind = torch.nonzero(self.reduce[index] <0)
            print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),len(filter_ind))
             
            for x in range(0, len(filter_ind)):
                self.fullbook[index][filter_ind[x]] = 0
      
        if ii == 2:
            length=ww.size()[0]*ww.size()[1]
            book=torch.ones(length,device=device)
            filter_ww = ww.view(-1)#.cpu().numpy()
            self.ww_delta[index]=(unit(filter_ww)*2-0.8)
            pos=torch.nonzero(self.ww_delta[index]>0)
            self.ww_delta[index][pos]=self.ww_delta[index][pos]+2
            self.reduceww[index]= self.reduceww[index]*0.999+self.ww_delta[index]*math.exp(-int((epoch-5)/15))
            filter_indww =torch.nonzero(self.reduceww[index] < 0)
            book[filter_indww]=0
            book=book.reshape((ww.size()[0],-1))
            self.fullbook[index]=self.fullbook[index]*book
            print(self.reduceww[index].mean(),self.reduceww[index].max(),self.reduceww[index].min(),len(filter_indww))
                
            wconv= dendrite#.cpu().numpy()
            self.n_delta[index]=(unit(wconv)*2-0.8)
            pos=torch.nonzero(self.n_delta[index]>0)
            self.n_delta[index][pos]=self.n_delta[index][pos]+2
            self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch-5)/15))
            filter_ind = torch.nonzero(self.reduce[index] <0)
            print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),len(filter_ind))
             
            for x in range(0, len(filter_ind)):
                self.fullbook[index][filter_ind[x]] = 0

        return self.fullbook[index]

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_mask(self, wwfc,convtra,epoch):
        for i in range(1,len(convlayer)):
            index=convlayer[i]
            ww = wwfc[index]
            dendrite=convtra[index]
            self.mat[index]=self.get_filter_codebook(ww, dendrite,4,index,epoch)
            #self.mat[index] = self.convert2tensor(self.mat[index]).cuda()
        for i in range(1,len(fclayer)):
            index=fclayer[i]
            ww=wwfc[index]
            dendrite=convtra[index]
            self.mat[index]=self.get_filter_codebook(ww,dendrite,2,index,epoch)
            #self.mat[index] = self.convert2tensor(self.mat[index]).cuda()

    def do_mask(self):
        for i in range(1,len(convlayer)):
            index=convlayer[i]
            ww = self.feature[index].weight
            maskww=ww*self.mat[index]
            self.feature[index].weight.data=maskww
        for i in range(1,len(fclayer)):
            ind=fclayer[i]
            l=fcc[i-1]
            ww = self.fc[l].weight
            maskww=ww*self.mat[ind]
            self.fc[l].weight.data=maskww

    def if_zero(self):
        cc=[]
        for i in range(1,len(convlayer)):
            ww=self.feature[convlayer[i]].weight
            b = ww.data.view(-1).cpu().numpy()
            print("number of weight is %d, zero is %.3f" %(len(b),100*(len(b)- np.count_nonzero(b))/len(b)))
            cc.append(100*(len(b)- np.count_nonzero(b))/len(b))
        for i in range(1,len(fcsize)):
            l=fcc[i-1]
            ww=self.fc[l].weight
            b = ww.data.view(-1).cpu().numpy()
            print("number of weight is %d, zero is %.3f" %(len(b),100*(len(b)- np.count_nonzero(b))/len(b)))
            cc.append(100*(len(b)- np.count_nonzero(b))/len(b))
        return cc            
            
#     def get_filter_codebook(self,dendrite,ii,index,epoch): 
#         if ii == 4:
#             wconv= dendrite.detach().cpu().numpy()
#             if index>15:
#                 self.n_delta[index]=(unit(wconv)*2-0.60)
#             elif index<15:
#                 self.n_delta[index]=(unit(wconv)*2-0.60)
#             pos=np.where(self.n_delta[index]>0)[0]
#             self.n_delta[index][pos]=self.n_delta[index][pos]+5
#             #print(wconv.mean(),wconv.max(), wconv.min())
#             self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch-5)/11))
#             filter_ind = np.where(self.reduce[index] <0)[0]
#             print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),len(filter_ind))
             
#             for x in range(0, len(filter_ind)):
#                 self.fullbook[index][filter_ind[x]] = 0
      
#         # if ii == 2:
#         #     length=ww.size()[0]*ww.size()[1]
#         #     book=torch.ones(length,device=device)
#         #     filter_ww = ww.view(-1)#.cpu().numpy()
#         #     self.ww_delta[index]=(unit(filter_ww)*2-0.65)
#         #     pos=torch.nonzero(self.ww_delta[index]>0)
#         #     self.ww_delta[index][pos]=self.ww_delta[index][pos]+2
#         #     self.reduceww[index]= self.reduceww[index]*0.999+self.ww_delta[index]*math.exp(-int((epoch-5)/9))
#         #     filter_indww =torch.nonzero(self.reduceww[index] < 0)
#         #     book[filter_indww]=0
#         #     book=book.reshape((ww.size()[0],-1))
#         #     self.fullbook[index]=self.fullbook[index]*book
#         #     print(self.reduceww[index].mean(),self.reduceww[index].max(),self.reduceww[index].min(),len(filter_indww))
                
#         #     wconv= dendrite#.cpu().numpy()
#         #     self.n_delta[index]=(unit(wconv)*2-0.65)
#         #     pos=torch.nonzero(self.n_delta[index]>0)
#         #     self.n_delta[index][pos]=self.n_delta[index][pos]+2
#         #     self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch-5)/9))
#         #     filter_ind = torch.nonzero(self.reduce[index] <0)
#         #     print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),len(filter_ind))
             
#         #     for x in range(0, len(filter_ind)):
#         #         self.fullbook[index][filter_ind[x]] = 0

#         return self.fullbook[index]

#     def convert2tensor(self, x):
#         x = torch.FloatTensor(x)
#         return x

#     def init_mask(self,convtra,epoch):
#         for i in range(layer):
#             if (i%3)!=0:
#                 dendrite=convtra[i]
#                 self.mat[i]=self.get_filter_codebook(dendrite,4,i,epoch)
#                 self.mat[i]=self.convert2tensor(self.mat[i]).cuda()
#             #self.mat[index] = self.convert2tensor(self.mat[index]).cuda()
#         # for i in range(1,len(fclayer)):
#         #     index=fclayer[i]
#         #     ww=wwfc[index]
#         #     dendrite=convtra[index]
#         #     self.mat[index]=self.get_filter_codebook(ww,dendrite,2,index,epoch)
#         #     #self.mat[index] = self.convert2tensor(self.mat[index]).cuda()

#     def do_mask(self):
#         n=0
#         for index, item in enumerate(self.model.parameters()):
#             if(index in self.mask_index and index!=0):
#                 ind=self.index[n]
#                 a = item.data
#                 #print(index,ind,a.size(),self.mat[ind].size())
#                 item.data = a * self.mat[ind]
#                 n=n+1
#         # for i in range(1,len(fclayer)):
#         #     ind=fclayer[i]
#         #     ww = self.fc[i].fc.weight
#         #     maskww=ww*self.mat[ind]
#         #     self.fc[i].fc.weight.data=maskww

#     def if_zero(self):
#         for index, item in enumerate(self.model.parameters()):
#             #print(index,item.data.size())
#             if(index in self.mask_index):
#                 a = item.data.view(-1)
#                 b = a.cpu().numpy()
                
#                 print("number of weight is %d, zero is %.3f" %(len(b),100*(len(b)- np.count_nonzero(b))/len(b)))
        
                

