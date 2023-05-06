import torch
import torch.nn as nn
import poly_utils as ut

class Legendre_test(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Legendre_test, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        self.T0 = nn.Parameter(torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(k,d)))

        #self.layer_C = nn.Parameter(ut.Norm_param(o,k))
        self.layer_C = nn.Linear(self.rank, self.output_dimension) 
        #self.beta = nn.Parameter(nn.Linear(o, 1).weight.to(torch.float32)[0])


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = torch.matmul(z, (self.T1).T)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            out0 = torch.matmul(z, ((2*i - 1)/(i))*getattr(self, 'T{}'.format(i)).T) * out1 - ((i - 1)/(i))*out0
            out_list = out_list + [out0]
            if i == self.degree:
                #x = torch.matmul(out_list[self.degree], self.layer_C.T) + self.beta
                x = self.layer_C(out_list[self.degree])
            else:
                j = i + 1
                out1 = torch.matmul(z, ((2*j - 1)/(j))*getattr(self, 'T{}'.format(i + 1)).T) * out0 - ((j - 1)/(j))*out1
                out_list = out_list + [out1]
        #x = torch.matmul(out_list[self.degree], self.layer_C.T) + self.beta
        x = self.layer_C(out_list[self.degree])
        return x

class Legendre(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Legendre, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        self.T0 = nn.Parameter(torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(k,d)))

        self.layer_C = nn.Parameter(ut.Norm_param(o,k))
        self.beta = nn.Parameter(nn.Linear(o, 1).weight.to(torch.float32)[0])


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = torch.matmul(z, (self.T1).T)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            out0 = torch.matmul(z, ((2*i - 1)/(i))*getattr(self, 'T{}'.format(i)).T) * out1 - ((i - 1)/(i))*out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = torch.matmul(out_list[self.degree], self.layer_C.T) 
                return x
            else:
                j = i + 1
                out1 = torch.matmul(z, ((2*j - 1)/(j))*getattr(self, 'T{}'.format(i + 1)).T) * out0 - ((j - 1)/(j))*out1
                out_list = out_list + [out1]
        x = torch.matmul(out_list[self.degree], self.layer_C.T) 
        return x
    
class Legendre0(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Legendre0, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        self.T0 = nn.Parameter(torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Linear(self.input_dimension, self.output_dimension, bias=False))
            #\setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(k,d)))

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 
        #self.beta = nn.Parameter(nn.Linear(o, 1).weight.to(torch.float32)[0])


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = self.T1(z)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            out0 = ((2*i - 1)/(i)) * getattr(self, 'T{}'.format(i))(z)  * out1 - ((i - 1)/(i)) * out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = self.layer_C(out_list[self.degree])
                return x
            else:
                j = i + 1
                out1 = ((2*j - 1)/(j)) * getattr(self, 'T{}'.format(j))(z)  * out1 - ((j - 1)/(j)) * out0
                out_list = out_list + [out1]
        x = self.layer_C(out_list[self.degree])
        return x
    
class Legendre1(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Legendre1, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        #self.T0 = nn.Parameter(torch.ones(k))
        self.register_buffer('T0', torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(k,d)))

        self.layer_C = nn.Parameter(ut.Norm_param(o,k))
        self.beta = nn.Parameter(nn.Linear(o, 1).weight.to(torch.float32)[0])


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = torch.matmul(z, (self.T1).T)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            out0 = torch.matmul(z, ((2*i - 1)/(i))*getattr(self, 'T{}'.format(i)).T) * out1 - ((i - 1)/(i))*out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = torch.matmul(out_list[self.degree], self.layer_C.T) + self.beta
                return x
            else:
                j = i + 1
                out1 = torch.matmul(z, ((2*j - 1)/(j))*getattr(self, 'T{}'.format(i + 1)).T) * out0 - ((j - 1)/(j))*out1
                out_list = out_list + [out1]
        x = torch.matmul(out_list[self.degree], self.layer_C.T) + self.beta
        return x
    
class Legendre1_5(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Legendre1_5, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        #self.T0 = nn.Parameter(torch.ones(k))
        self.register_buffer('T0', torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(k,d)))

        self.layer_C = nn.Parameter(ut.Norm_param(o,k))
        #self.beta = nn.Parameter(nn.Linear(o, 1).weight.to(torch.float32)[0])


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = torch.matmul(z, (self.T1).T)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            out0 = torch.matmul(z, ((2*i - 1)/(i))*getattr(self, 'T{}'.format(i)).T) * out1 - ((i - 1)/(i))*out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = torch.matmul(out_list[self.degree], self.layer_C.T) 
                return x
            else:
                j = i + 1
                out1 = torch.matmul(z, ((2*j - 1)/(j))*getattr(self, 'T{}'.format(i + 1)).T) * out0 - ((j - 1)/(j))*out1
                out_list = out_list + [out1]
        x = torch.matmul(out_list[self.degree], self.layer_C.T) 
        return x
    
class Legendre2(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Legendre2, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        #self.T0 = nn.Parameter(torch.ones(k))
        self.register_buffer('T0', torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(k,d)))

        self.layer_C = nn.Parameter(ut.Norm_param(o,k))
        self.beta = nn.Parameter(nn.Linear(o, 1).weight.to(torch.float32)[0])


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = torch.matmul(z, (self.T1).T)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            Ti = getattr(self, 'T{}'.format(i))
            Ti = Ti/(torch.sum(Ti, dim=1)).reshape(-1,1) 
            out0 = torch.matmul(z, ((2*i - 1)/(i))*Ti.T) * out1 - ((i - 1)/(i))*out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = torch.matmul(out_list[self.degree], self.layer_C.T) + self.beta
                return x
            else:
                j = i + 1
                Tj = getattr(self, 'T{}'.format(j))
                Tj = Tj/(torch.sum(Tj, dim=1)).reshape(-1,1)
                out1 = torch.matmul(z, ((2*j - 1)/(j))*Tj.T) * out0 - ((j - 1)/(j))*out1
                out_list = out_list + [out1]
        x = torch.matmul(out_list[self.degree], self.layer_C.T) + self.beta
        return x
    
class Legendre3(nn.Module):
    def __init__(self, degree, d, k, o):
        super(Legendre3, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        self.T0 = nn.Parameter(torch.ones(k))
        #self.register_buffer('T0', torch.ones(k))
        for i in range(1, self.degree + 1):
            setattr(self, 'T{}'.format(i), nn.Parameter(ut.Norm_param(k,d)))

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 

    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out0 = self.T0
        out1 = torch.matmul(z, (self.T1).T)
        out_list = [out0] + [out1]
        for i in range(2, self.degree + 1, 2):
            Ti = getattr(self, 'T{}'.format(i))
            Ti = Ti/(torch.sum(Ti, dim=1)).reshape(-1,1) 
            out0 = torch.matmul(z, ((2*i - 1)/(i))*Ti.T) * out1 - ((i - 1)/(i))*out0
            out_list = out_list + [out0]
            if i == self.degree:
                x = self.layer_C(out_list[self.degree])
                return x
            else:
                j = i + 1
                Tj = getattr(self, 'T{}'.format(j))
                Tj = Tj/(torch.sum(Tj, dim=1)).reshape(-1,1)
                out1 = torch.matmul(z, ((2*j - 1)/(j))*Tj.T) * out0 - ((j - 1)/(j))*out1
                out_list = out_list + [out1]
        x = self.layer_C(out_list[self.degree])
        return x