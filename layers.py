import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class GraphConvolution(Module):
    def __init__(self,input_dim,output_dim,bias=True):
        super(GraphConvolution,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim,output_dim))

        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1/math.sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight,gain=1.414)
        if self.bias is not None:
            nn.init.uniform_(self.bias,-stdv,stdv)

    def forward(self,x,adj):
        x = torch.mm(x,self.weight)
        output = torch.mm(adj,x)
        if self.bias is not None:
            output = output+self.bias
        return output
