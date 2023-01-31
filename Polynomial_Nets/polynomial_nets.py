import torch
import torch.nn as nn


class NCP_L3(nn.Module):
  def __init__(self, d, k, o, w):
    super(NCP_L3, self).__init__()
    self.layer_b1 = torch.nn.Parameter(torch.randn(w))
    self.layer_b2 = torch.nn.Parameter(torch.randn(w))
    self.layer_b3 = torch.nn.Parameter(torch.randn(w))
    self.layer_A1 = nn.Linear(d, k, bias=False)
    self.layer_A2 = nn.Linear(d, k, bias=False)
    self.layer_A3 = nn.Linear(d, k, bias=False)
    self.layer_B1 = nn.Linear(w, k, bias=False)
    self.layer_B2 = nn.Linear(w, k, bias=False)
    self.layer_B3 = nn.Linear(w, k, bias=False)
    self.layer_S2 = nn.Linear(k, k, bias=False)
    self.layer_S3 = nn.Linear(k, k, bias=False)
    self.layer_C = nn.Linear(k, o)
   #[-1, 1] 


  def forward(self, z):
    x1 = self.layer_A1(z)*self.layer_B1(self.layer_b1)
    x2 = self.layer_A2(z)*(self.layer_S2(x1) + self.layer_B2(self.layer_b2))
    x3 = self.layer_A3(z)*(self.layer_S3(x2) + self.layer_B3(self.layer_b3))
    x = self.layer_C(x3)
   
    return x

class CP_L3(nn.Module):
    def __init__(self, d, k, o):
        super(CP_L3, self).__init__()
        
        self.layer_U1 = nn.Linear(d, k, bias=False)
        self.layer_U2 = nn.Linear(d, k, bias=False)
        self.layer_U3 = nn.Linear(d, k, bias=False)
        self.layer_C = nn.Linear(k, o)   
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        x1 = self.layer_U1(z)
        x2 = self.layer_U2(z)*x1 + x1 
        x3 = self.layer_U3(z)*x2 + x2 
        x = self.layer_C(x3)
        return x

class CP_L3_sparse(nn.Module):
    def __init__(self, d, k, o):
        super(CP_L3_sparse, self).__init__()
        
        self.layer_U1 = nn.Parameter(torch.randn(k, d))  
        self.layer_U2 = nn.Parameter(torch.tril(torch.randn(k, d)))  
        self.layer_U3 = nn.Parameter(torch.tril(torch.randn(k, d)))  
        self.mask = torch.tril(torch.ones_like(self.layer_U2))
        self.layer_C = nn.Linear(k, o)   
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        x1 = torch.matmul(z, self.layer_U1.T)
        x2 = torch.matmul(z, (self.mask * self.layer_U2).T) + x1
        x3 = torch.matmul(z, (self.mask * self.layer_U3).T) + x2
        x = self.layer_C(x3)
        return x

class CP_L3_sparse_1(nn.Module):
    def __init__(self, d, k, o):
        super(CP_L3_sparse_1, self).__init__()
        
        self.layer_U1 = nn.Parameter(torch.tril(torch.randn(k, d)))     
        self.layer_U2 = nn.Parameter(torch.tril(torch.randn(k, d)))  
        self.layer_U3 = nn.Parameter(torch.tril(torch.randn(k, d)))  
        self.mask = torch.tril(torch.ones_like(self.layer_U1))
        self.layer_C = nn.Linear(k, o)   
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        x1 = torch.matmul(z, (self.mask * self.layer_U1).T)
        x2 = torch.matmul(z, (self.mask * self.layer_U2).T) + x1
        x3 = torch.matmul(z, (self.mask * self.layer_U3).T) + x2
        x = self.layer_C(x3)
        return x

class CP_L3_sparse_outer(nn.Module):
    def __init__(self, d, o):
        super(CP_L3_sparse_outer, self).__init__()
        
        self.layer_U1 = nn.Parameter(torch.randn(d))  
        self.layer_U2 = nn.Parameter(torch.randn(d))
        self.layer_U3 = nn.Parameter(torch.randn(d))
        self.layer_C = nn.Linear(d, o)   
        self.input_dimension = d 
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        x1 = self.layer_U1 * z
        x2 = self.layer_U2 * z
        x3 = self.layer_U3 * z
        x_outer1 = torch.outer(x2, x1)
        x_12 = torch.sum(x_outer1, 1)
        x_outer2 = torch.outer(x3, x_12)
        x_123 = torch.sum(x_outer2, 1)
        x = self.layer_C(x_123)
        return x