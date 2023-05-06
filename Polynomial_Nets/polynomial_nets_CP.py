import torch
import torch.nn as nn
import poly_utils as ut

class CP(nn.Module):
    def __init__(self, degree, d, k, o):
        super(CP, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        for i in range(1, self.degree + 1):
            setattr(self, 'U{}'.format(i), nn.Linear(self.input_dimension, self.rank, bias=False)) 

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out = self.U1(z)
        for i in range(2, self.degree + 1):
            out = getattr(self, 'U{}'.format(i))(z) * out + out
        x = self.layer_C(out)
        return x
    
class CP_sparse_LU(nn.Module):
    def __init__(self, degree, d, k, o, l_offset = 0, u_offset = 0):
        super(CP_sparse_LU, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        if self.input_dimension > self.rank:
            self.register_buffer('mask1', torch.triu(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = u_offset))
            self.register_buffer('mask2', torch.tril(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = l_offset))
            for i in range(1, self.degree + 1, 2):
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.triu(nn.Linear(d, k, bias=False).weight, diagonal = u_offset))) 
            for i in range(2, self.degree + 1, 2):
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight, diagonal = l_offset)))  
        else:
            self.register_buffer('mask1', torch.tril(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = l_offset))
            self.register_buffer('mask2', torch.triu(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = u_offset))

            for i in range(1, self.degree + 1, 2):
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight, diagonal = l_offset))) 
            for i in range(2, self.degree + 1, 2):
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.triu(nn.Linear(d, k, bias=False).weight, diagonal = u_offset)))         

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask1 * self.U1).T)
        for i in range(2, self.degree + 1):
            #out = getattr(self, 'U{}'.format(i))(z) * out + out
            out = torch.matmul(z, (self.mask2 * getattr(self, 'U{}'.format(i))).T) * out + out
            if i == self.degree:
                x = self.layer_C(out)
                return x
            out = torch.matmul(z, (self.mask1 * getattr(self, 'U{}'.format(i+1))).T) * out + out
        x = self.layer_C(out)
        return x

class CP_sparse_degree(nn.Module):
    def __init__(self, degree, d, k, o):
        super(CP_sparse_degree, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        masks = ut.generate_masks(self.degree, self.rank, self.input_dimension)
        for i in range(1, self.degree + 1):
            setattr(self, 'U{}'.format(i), nn.Parameter(nn.Linear(d, k, bias=False).weight * masks[i-1]))
            self.register_buffer('mask{}'.format(i), masks[i-1])        

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask1 * self.U1).T)
        for i in range(2, self.degree + 1):
            #out = getattr(self, 'U{}'.format(i))(z) * out + out
            out = torch.matmul(z, (getattr(self, 'mask{}'.format(i)) * getattr(self, 'U{}'.format(i))).T) * out + out
        x = self.layer_C(out)
        return x
    
class CP_sparse_L(nn.Module):
    def __init__(self, degree, d, k, o):
        super(CP_sparse_L, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        for i in range(1, self.degree + 1):
            setattr(self, 'U{}'.format(i), nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight))) 

        self.register_buffer('mask', torch.tril(torch.ones_like(self.U2)))

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask * self.U1).T)
        for i in range(2, self.degree + 1):
            #out = getattr(self, 'U{}'.format(i))(z) * out + out
            out = torch.matmul(z, (self.mask * getattr(self, 'U{}'.format(i))).T) * out + out
        x = self.layer_C(out)
        return x
    
class CP_sparse_U(nn.Module):
    def __init__(self, degree, d, k, o):
        super(CP_sparse_U, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        for i in range(1, self.degree + 1):
            setattr(self, 'U{}'.format(i), nn.Parameter(torch.triu(nn.Linear(d, k, bias=False).weight))) 

        self.register_buffer('mask', torch.triu(torch.ones_like(self.U2)))

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask * self.U1).T)
        for i in range(2, self.degree + 1):
            #out = getattr(self, 'U{}'.format(i))(z) * out + out
            out = torch.matmul(z, (self.mask * getattr(self, 'U{}'.format(i))).T) * out + out
        x = self.layer_C(out)
        return x