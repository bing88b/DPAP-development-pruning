import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
convlayer = [0, 1, 3, 4, 6, 7]
fclayer=[9,10]
imgsize = [64, 64, 32, 32, 16, 16]
batch = 4
step = 32
delta = 0.5
size = [2,128, 128, 256, 256, 512,512]
fcsize=[512*8*8,512]
#fcsize=[512*8*8,512]#,100]
'''convlayer = [0, 2, 4, 6]
fclayer=[7,8,9]
imgsize = [64, 32, 16,8]
batch = 20
step = 16
delta = 0.5
size = [2,128, 256, 256, 512]
fcsize=[512*8*8,512,100]'''


class trace:
    def __init__(self, model):
        self.model = model
        self.feature=model.feature
        self.ctrace={}
        self.fctrace={}
        self.csum={}
        self.fcsum={}
        

    def init(self):
        for i in range(len(convlayer)):
            self.ctrace[i]=torch.zeros((batch,size[i+1],imgsize[i],imgsize[i])).cuda()
        for i in range(len(fclayer)):
            self.fctrace[i]=torch.zeros((batch,fcsize[i])).cuda()
 

    def computing_trace(self,spikes):
        for i in range(len(convlayer)):
            self.ctrace[i]=torch.zeros((batch,size[i+1],imgsize[i],imgsize[i]),device=device)
        for i in range(len(fclayer)):
            self.fctrace[i]=torch.zeros((batch,fcsize[i]),device=device)
        for t in range(step):      
            for i in range(len(convlayer)):
                index=convlayer[i]
                sp=spikes[t][index].detach()
                #print(sp.size())
                self.ctrace[i]=delta*self.ctrace[i].cuda()+sp.cuda()
            for i in range(len(fclayer)):
                ind=fclayer[i]
                sp=spikes[t][ind].detach()
                self.fctrace[i]=delta*self.fctrace[i].cuda()+sp.cuda()
        for i in range(len(convlayer)):
            self.csum[i]=torch.sum(self.ctrace[i],dim=0)
            self.csum[i]=self.csum[i]/(step*batch)
        for i in range(len(fclayer)):
            self.fcsum[i]=torch.sum(self.fctrace[i],dim=0)
            self.fcsum[i]=self.fcsum[i]/(step*batch)
        return self.csum,self.fcsum,self.fctrace
