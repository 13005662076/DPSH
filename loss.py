import torch
import torch.nn as nn
from torch.autograd import Variable

class dpshloss(nn.Module):
    def __init__(self,num_train,bit):
        super(dpshloss,self).__init__()
        #定义输出[-1,1]矩阵
        self.B=torch.zeros(num_train,bit)
        #定义输出矩阵
        self.U=torch.zeros(num_train,bit)

    def forward(self,output,S,batch_index,num_train,num_label):
        for i,ind in enumerate(batch_index):
            self.U[ind,:]=output.data[i]
            self.B[ind,:]=torch.sign(output.data[i])
        Bb=torch.sign(output)
        if torch.cuda.is_available():
            x=output.mm(Variable(self.U.cuda()).t())/2
            loss1=(Variable(S.cuda())*x-torch.log(1+torch.exp(x))).sum()/(num_train*num_label)
            loss2=(Bb-output).pow(2).sum()/(num_train*num_label)
        else:
            x=output.mm(Variable(self.U).t())/2
            loss1=(Variable(S)*x-torch.log(1+torch.exp(x))).sum()/(num_train*num_label)
            loss2=(Bb-output).pow(2).sum()/(num_train*num_label)
        return loss1,loss2
