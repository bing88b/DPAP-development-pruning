import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
convlayer = [-1,0, 1, 3, 4, 6, 7]
fclayer=[8,9]
imgsize = [32,32, 32, 16,16, 16, 8,8, 8]
#imgsize = [64,64, 64, 32,32, 32, 16,16, 16]
batch = 50
step = 8
delta = 0.5
size = [3,128, 128, 256, 256, 512,512]
sizee = [3,128, 128,128, 256, 256,256, 512,512]
fcsize=[512*8*8,512]



class trace:
    def __init__(self, model):
        self.model = model
        self.feature=model.feature
        self.ctrace={}
        self.fctrace={}
        self.csum={}
        self.fcsum={}
        

    def computing_trace(self,spikes):
        for i in range(len(imgsize)):
            index=i-1
            self.ctrace[index]=torch.zeros((batch,sizee[i],imgsize[i],imgsize[i]),device=device)
        for i in range(len(fclayer)):
            index=fclayer[i]
            self.fctrace[index]=torch.zeros((batch,fcsize[i]),device=device)
        for t in range(step):      
            for i in range(len(imgsize)):
                index=i-1
                sp=spikes[t][index+1].detach()
                #print(sp.size(),self.ctrace[index].size())
                self.ctrace[index]=delta*self.ctrace[index].cuda()+sp.cuda()
            for i in range(len(fclayer)):
                index=fclayer[i]
                sp=spikes[t][index+1].detach()
                self.fctrace[index]=delta*self.fctrace[index].cuda()+sp.cuda()
        for i in range(len(imgsize)):
            index=i-1
            self.csum[index]=self.ctrace[index]/(step)
            self.csum[index]=torch.sum(torch.sum(self.csum[index],dim=2),dim=2)
        for i in range(len(fclayer)):
            index=fclayer[i]
            self.fcsum[index]=self.fctrace[index]/(step)
        return self.csum,self.fcsum
