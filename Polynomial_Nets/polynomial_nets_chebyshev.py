import torch
import torch.nn as nn
import poly_utils as ut

'''class Chebyshev2(nn.Module):
    def __init__(self, degree, d, o):
        super(Chebyshev2, self).__init__()        
     
        self.input_dimension = d 
        #self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        self.T0 = nn.Parameter(nn.Linear(o, 1).weight.to(torch.float32)[0])
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Linear(self.input_dimension, self.output_dimension, bias=False)) 

        #self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = self.T1(z)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            out0 = getattr(self, 'T{}'.format(i))(z) * 2 * out1 - out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = out_list[self.degree]
                return x
            else:
                out1 = getattr(self, 'T{}'.format(i + 1))(z) * 2 * out0 - out1
                out_list = out_list + [out1]
        x = out_list[self.degree]
        return x'''
    
class Chebyshev(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Chebyshev, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        self.T0 = nn.Parameter(nn.Linear(k, 1).weight.to(torch.float32)[0])
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Linear(self.input_dimension, self.rank, bias=False)) 

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = self.T1(z)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            out0 = getattr(self, 'T{}'.format(i))(z) * 2 * out1 - out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = self.layer_C(out_list[self.degree])
                return x
            else:
                out1 = getattr(self, 'T{}'.format(i + 1))(z) * 2 * out0 - out1
                out_list = out_list + [out1]
        x = self.layer_C(out_list[self.degree])
        return x
    
class Chebyshev_norm(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Chebyshev_norm, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        self.T0 = nn.Parameter(torch.ones(k))
        #self.register_buffer('T0', torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(k,d)))

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 
        #self.beta = nn.Parameter(nn.Linear(o, 1).weight.to(torch.float32)[0])


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = torch.matmul(z, (self.T1).T)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            Ti = getattr(self, 'T{}'.format(i))
            Ti = Ti/(torch.sum(Ti, dim=1)).reshape(-1,1)                     
            out0 = torch.matmul(z, 2*Ti.T) * out1 - out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = self.layer_C(out_list[self.degree])
                return x
            else:
                Ti = getattr(self, 'T{}'.format(i + 1))
                Ti = Ti/(torch.sum(Ti, dim=1)).reshape(-1,1)
                out1 = torch.matmul(z, 2*Ti.T) * out0 - out1
                out_list = out_list + [out1]
        x = self.layer_C(out_list[self.degree])
        return x
    
class Chebyshev_test(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Chebyshev_test, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        self.T0 = nn.Parameter(torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(self.rank, self.input_dimension))) 

        self.layer_C = nn.Parameter(ut.Norm_param(o,k)) 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = torch.matmul(z, (self.T1).T)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            out0 = torch.matmul(z, 2*getattr(self, 'T{}'.format(i)).T) * out1 - out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = torch.matmul(out_list[self.degree], self.layer_C.T)
                return x
            else:
                out1 = torch.matmul(z, 2*getattr(self, 'T{}'.format(i+1)).T) * out0 - out1
                out_list = out_list + [out1]
        x = torch.matmul(out_list[self.degree], self.layer_C.T)
        return x