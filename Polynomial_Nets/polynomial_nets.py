import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

def weight_matrix_k4s2(image_size):
    
    kernel_width = 16
    kernel_height = kernel_width
    image_width = image_size
    image_height = image_width
    stride = 2
    width_steps = int((image_width - kernel_width)/stride)
    height_steps = int((image_height - kernel_height)/stride)

    base = torch.arange(0, kernel_width)
    base_new = base.repeat(kernel_height)
    addition = torch.arange(0, kernel_width) * image_width
    addition_new  = addition.repeat_interleave(kernel_height)
    index_1 = addition_new + base_new

    index_width = index_1.repeat(width_steps + 1)
    addition_width = stride * torch.arange(0, width_steps + 1)
    addition_width = addition_width.repeat_interleave(kernel_height * kernel_width)
    index_row = index_width + addition_width

    index_column = index_row.repeat(height_steps + 1)
    addition_height = stride * image_width * torch.arange(0, height_steps + 1)
    addition_height_rep = addition_height.repeat_interleave(kernel_height * kernel_width * (height_steps + 1))
    index_final = index_column + addition_height_rep

    stack = torch.arange(0, (height_steps + 1) * (width_steps + 1)).repeat_interleave(kernel_height * kernel_width)
    indices = torch.stack((stack, index_final), dim=1)

    W_in = image_height * image_width
    W_out = (height_steps + 1) * (width_steps + 1)
    #print(W_out, W_in)
    W = torch.zeros([W_out, W_in], dtype=torch.float32)
    values = torch.nn.Linear(indices.shape[0], 1).weight.to(torch.float32)[0]
    W = W.index_put_(tuple(indices.t()), values)

    mask = torch.zeros([W_out, W_in], dtype=torch.float32)
    values_m = torch.ones_like(values)
    mask = mask.index_put_(tuple(indices.t()), values_m)

    return W, mask

class Attention(nn.Module):
    def __init__(self, d, k, o):
        super(Attention, self).__init__()
        
        self.layer_U_q = nn.Linear(d, k, bias=False)
        self.layer_U_k = nn.Linear(d, k, bias=False)
        self.layer_U_v = nn.Linear(d, k, bias=False)

        self.layer_C = nn.Linear(k, o)   
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        q = self.layer_U_q(z)
        k = self.layer_U_k(z)
        v = self.layer_U_v(z)
        energy = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) / self.rank ** 0.5
        attention = energy.softmax(dim = -1)
        out = torch.bmm(v.unsqueeze(1), attention.permute(0,2,1))
        
        x = self.layer_C(out)
        return x

class Chebyshev_L3_sparseD32_stack(nn.Module):
    def __init__(self, d, o):
        super(Chebyshev_L3_sparseD32_stack, self).__init__()        
        
        self.layer_T1 = nn.Parameter(torch.vstack([weight_matrix_k4s2(d)[0]]*10)).to(device)        
        self.layer_T2 = nn.Parameter(torch.vstack([weight_matrix_k4s2(d)[0]]*10)).to(device) 
        self.layer_T3 = nn.Parameter(torch.vstack([weight_matrix_k4s2(d)[0]]*10)).to(device) 
        #self.layer_T4 = nn.Parameter(weight_matrix_k4s2(d)[0]).to(device) 
        self.mask = torch.vstack([weight_matrix_k4s2(d)[1]]*10).to(device)
        self.k = self.layer_T2.shape[0]
        self.layer_T0 = torch.nn.Parameter(torch.randn(self.k))
        #self.layer_T1 = nn.Linear(d * d, self.k, bias=False).weight

        self.layer_C = nn.Linear(self.k, o)   
        self.input_dimension = d 
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension * self.input_dimension)
        x0 = self.layer_T0 
        x1 = torch.matmul(z, (self.mask * self.layer_T1).T)
        x2 = (2 * torch.matmul(z, (self.mask * self.layer_T2).T) * x1) - x0
        x3 = (2 * torch.matmul(z, (self.mask * self.layer_T3).T) * x2) - x1 
        #x4 = (2 * torch.matmul(z, (self.mask * self.layer_T4).T) * x3) - x2  
        x = self.layer_C(x3)
        return x

class Chebyshev_L3_sparseD32(nn.Module):
    def __init__(self, d, o):
        super(Chebyshev_L3_sparseD32, self).__init__()        
        
        self.layer_T1 = nn.Parameter(weight_matrix_k4s2(d)[0]).to(device)        
        self.layer_T2 = nn.Parameter(weight_matrix_k4s2(d)[0]).to(device) 
        self.layer_T3 = nn.Parameter(weight_matrix_k4s2(d)[0]).to(device) 
        self.layer_T4 = nn.Parameter(weight_matrix_k4s2(d)[0]).to(device) 
        self.mask = weight_matrix_k4s2(d)[1].to(device)
        self.k = self.layer_T2.shape[0]
        self.layer_T0 = torch.nn.Parameter(torch.randn(self.k))
        #self.layer_T1 = nn.Linear(d * d, self.k, bias=False).weight

        self.layer_C = nn.Linear(self.k, o)   
        self.input_dimension = d 
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension * self.input_dimension)
        x0 = self.layer_T0 
        x1 = torch.matmul(z, (self.mask * self.layer_T1).T)
        x2 = (2 * torch.matmul(z, (self.mask * self.layer_T2).T) * x1) - x0
        x3 = (2 * torch.matmul(z, (self.mask * self.layer_T3).T) * x2) - x1 
        x4 = (2 * torch.matmul(z, (self.mask * self.layer_T4).T) * x3) - x2  
        x = self.layer_C(x4)
        return x
    
class Chebyshev_sparse_kernelD(nn.Module):
    def __init__(self, d, o):
        super(Chebyshev_sparse_kernelD, self).__init__()        
        
        #self.layer_T1 = nn.Parameter(weight_matrix_k4s2(d)[0]).to(device)        
        self.layer_T2 = nn.Parameter(torch.vstack([weight_matrix_k4s2(d)[0]]*5)).to(device) 
        self.layer_T3 = nn.Parameter(torch.vstack([weight_matrix_k4s2(d)[0]]*5)).to(device)
        #self.layer_T4 = nn.Parameter(weight_matrix_k4s2(d)[0]).to(device) 
        self.mask = torch.vstack([weight_matrix_k4s2(d)[1]]*5).to(device)
        self.k = self.layer_T2.shape[0]
        self.layer_T0 = torch.nn.Parameter(torch.randn(self.k))
        self.layer_T1 = nn.Linear(d * d, self.k, bias=False)

        self.layer_C = nn.Linear(self.k, o)   
        self.input_dimension = d 
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension * self.input_dimension)
        x0 = self.layer_T0 
        x1 = self.layer_T1(z)
        x2 = (2 * torch.matmul(z, (self.mask * self.layer_T2).T) * x1) - x0
        x3 = (2 * torch.matmul(z, (self.mask * self.layer_T3).T) * x2) - x1 
        #x3 = (2 * torch.matmul(z, (self.mask * self.layer_T4).T) * x3) - x2  
        x = self.layer_C(x3)
        return x
    
class Chebyshev_sparse_kernelU(nn.Module):
    def __init__(self, d, o):
        super(Chebyshev_sparse_kernelU, self).__init__()        
        
        #self.layer_T1 = nn.Parameter(weight_matrix_k4s2(o)[0].T)
        self.layer_T2 = nn.Parameter(weight_matrix_k4s2(o)[0].T)
        self.layer_T3 = nn.Parameter(weight_matrix_k4s2(o)[0].T)
        #self.layer_T4 = nn.Parameter(weight_matrix_k4s2(o)[0].T)
        self.mask = weight_matrix_k4s2(o)[1].T.to(device)
        self.k = self.layer_T2.shape[1]
        self.layer_T0 = nn.Linear(o * o, 1).weight.to(torch.float32)[0].to(device)
        self.layer_T1 = nn.Linear(self.k, o * o, bias=False)

        self.layer_C = nn.Linear(d, self.k)   
        self.input_dimension = d 
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        z = self.layer_C(z)
        x0 = self.layer_T0 
        #print(self.mask.shape, self.layer_T1.shape)
        x1 = self.layer_T1(z)
        #print(z.device, self.mask.device, self.layer_T2.device, x0.device, x1.device)
        x2 = (2 * torch.matmul(z, (self.mask * self.layer_T2).T) * x1) - x0
        x = (2 * torch.matmul(z, (self.mask * self.layer_T3).T) * x2) - x1 
        #x = (2 * torch.matmul(z, (self.mask * self.layer_T4).T) * x3) - x2  
        #print(x.dtype)       
        return x
    
class Chebyshev_L3_sparseU32(nn.Module):
    def __init__(self, d, o):
        super(Chebyshev_L3_sparseU32, self).__init__()        
        
        self.layer_T1 = nn.Parameter(weight_matrix_k4s2(o)[0].T)
        self.layer_T2 = nn.Parameter(weight_matrix_k4s2(o)[0].T)
        self.layer_T3 = nn.Parameter(weight_matrix_k4s2(o)[0].T)
        self.mask = weight_matrix_k4s2(o)[1].T.to(device)
        self.k = self.layer_T2.shape[1]
        self.layer_T0 = nn.Linear(o * o, 1).weight.to(torch.float32)[0].to(device)
        #self.layer_T1 = nn.Linear(self.k, o * o, bias=False).weight

        self.layer_C = nn.Linear(d, self.k)   
        self.input_dimension = d 
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        z = self.layer_C(z)
        x0 = self.layer_T0 
        #print(self.mask.shape, self.layer_T1.shape)
        x1 = torch.matmul(z, (self.mask * self.layer_T1).T)
        #print(z.device, self.mask.device, self.layer_T2.device, x0.device, x1.device)
        x2 = (2 * torch.matmul(z, (self.mask * self.layer_T2).T) * x1) - x0
        x = (2 * torch.matmul(z, (self.mask * self.layer_T3).T) * x2) - x1  
        #print(x.dtype)       
        return x


class Chebyshev_L3(nn.Module):
    def __init__(self, d, k, o):
        super(Chebyshev_L3, self).__init__()
        
        self.layer_a = torch.nn.Parameter(torch.randn(k))
        self.layer_b = nn.Linear(d, k, bias=False)
        self.layer_c = nn.Linear(d, k, bias=False)
        self.layer_d = nn.Linear(d, k, bias=False)

        self.layer_C = nn.Linear(k, o)   
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 


    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        T0 = self.layer_a 
        T1 = self.layer_b(z)
        T2 = 2 * self.layer_c(z) * T1 - T0
        T3 = 2 * self.layer_d(z) * T2 - T1
        #T4 = 2 * self.layer_b(z) * T3 - T2
        #T5 = 2 * self.layer_b(z) * T4 - T3
        
        x = self.layer_C(T3)
        return x

class Chebyshev_L3_sparse(nn.Module):
    def __init__(self, d, k, o):
        super(Chebyshev_L3_sparse, self).__init__()

        self.layer_a = torch.nn.Parameter(torch.randn(k))
        self.layer_b = nn.Linear(d, k, bias=False)
        self.layer_c = nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight))
        self.layer_d = nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight))
        self.mask = torch.tril(torch.ones_like(self.layer_c)).to(device)        
        self.layer_C = nn.Linear(k, o)   
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 

    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        T0 = self.layer_a
        T1 = self.layer_b(z)
        T2 = (2 * torch.matmul(z, (self.mask * self.layer_c).T) * T1) - T0
        T3 = (2 * torch.matmul(z, (self.mask * self.layer_d).T) * T2) - T1
        x = self.layer_C(T3)
        return x
    
class ProdCheby(nn.Module):
    def __init__(self, layer_params = [], degrees = [], output = 1):
        super(ProdCheby, self).__init__()   
        
        self.input_dimension = layer_params[0]
        self.output_dimension = output
        self.num_layers = len(degrees)
        for i in range(self.num_layers):
            setattr(self, 'layer{}'.format(i+1), Chebyshev2(degrees[i], layer_params[i], layer_params[i+1])) 
        self.layer_C = nn.Linear(layer_params[self.num_layers], self.output_dimension)

    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        for i in range(self.num_layers):
            z = getattr(self, 'layer{}'.format(i+1))(z)
        x = self.layer_C(z)    
        return x
    
class ProdCheby2(nn.Module):
    def __init__(self, layer_params = [], degrees = [], output = 1):
        super(ProdCheby2, self).__init__()   
        
        self.input_dimension = layer_params[0]
        self.output_dimension = output
        self.num_layers = len(degrees)
        for i in range(self.num_layers-1):
            setattr(self, 'layer{}'.format(i+1), Chebyshev(degrees[i], layer_params[i], layer_params[i+1], layer_params[i+1])) 
        setattr(self, 'layer{}'.format(self.num_layers), Chebyshev(degrees[self.num_layers-1], layer_params[self.num_layers-1], layer_params[self.num_layers], self.output_dimension))
        #self.layer_C = nn.Linear(layer_params[self.num_layers], self.output_dimension)

    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        for i in range(self.num_layers):
            z = getattr(self, 'layer{}'.format(i+1))(z)
        #x = self.layer_C(z)    
        return z
    
class ProdCheby_NL(nn.Module):
    def __init__(self, layer_params = [], degrees = [], output = 1, activation = nn.ReLU()):
        super(ProdCheby_NL, self).__init__()   
        
        self.input_dimension = layer_params[0]
        self.output_dimension = output
        self.num_layers = len(degrees)
        self.activation = activation
        for i in range(self.num_layers-1):
            setattr(self, 'layer{}'.format(i+1), Chebyshev(degrees[i], layer_params[i], layer_params[i+1], layer_params[i+1])) 
        setattr(self, 'layer{}'.format(self.num_layers), Chebyshev(degrees[self.num_layers-1], layer_params[self.num_layers-1], layer_params[self.num_layers], self.output_dimension))
        #self.layer_C = nn.Linear(layer_params[self.num_layers], self.output_dimension)

    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        for i in range(self.num_layers-1):
            z = self.activation(getattr(self, 'layer{}'.format(i+1))(z))
        z = getattr(self, 'layer{}'.format(self.num_layers))(z)
            
        return z
    
class ProdCheby_NL2(nn.Module):
    def __init__(self, layer_params = [], degrees = [], output = 1,  activation = nn.ReLU()):
        super(ProdCheby_NL2, self).__init__()   
        
        self.input_dimension = layer_params[0]
        self.output_dimension = output
        self.num_layers = len(degrees)
        self.activation = activation
        for i in range(self.num_layers):
            setattr(self, 'layer{}'.format(i+1), Chebyshev2(degrees[i], layer_params[i], layer_params[i+1])) 
        self.layer_C = nn.Linear(layer_params[self.num_layers], self.output_dimension)

    def forward(self, z):
        z = z.reshape(-1, self.input_dimension)
        for i in range(self.num_layers):
            z = self.activation(getattr(self, 'layer{}'.format(i+1))(z))
        x = self.layer_C(z)    
        return x
    

    
class Chebyshev2(nn.Module):
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
        return x
    
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


class NCP_L3_skip(nn.Module):
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
    x1 = self.layer_A1(z) * self.layer_B1(self.layer_b1)
    x2 = self.layer_A2(z) * (self.layer_S2(x1) + self.layer_B2(self.layer_b2)) + x1
    x3 = self.layer_A3(z) * (self.layer_S3(x2) + self.layer_B3(self.layer_b3)) + x2
    x = self.layer_C(x3)
   
    return x

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
    x1 = self.layer_A1(z) * self.layer_B1(self.layer_b1)
    x2 = self.layer_A2(z) * (self.layer_S2(x1) + self.layer_B2(self.layer_b2))
    x3 = self.layer_A3(z) * (self.layer_S3(x2) + self.layer_B3(self.layer_b3))
    x = self.layer_C(x3)
   
    return x
  
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

class CP_C3(nn.Module):
  def __init__(self, d, k, o):
    super(CP_C3, self).__init__()
    self.layer_U1 = nn.Conv2d(d, k, kernel_size=4, stride=2, padding=2, bias=False)
    self.layer_U2 = nn.Conv2d(d, k, kernel_size=4, stride=2, padding=2, bias=False)
    self.layer_U3 = nn.Conv2d(d, k, kernel_size=4, stride=2, padding=2, bias=False)
    self.layer_C = nn.Conv2d(k, o, kernel_size=4, stride=2, padding=2, bias=True)
   #[-1, 1] 


  def forward(self, z):
     x1 = self.layer_U1(z)
     x2 = self.layer_U2(z) * x1 + x1 
     x3 = self.layer_U3(z) * x2 + x2 
     x = self.layer_C(x3)
     return x

class CP_CT3(nn.Module):
  def __init__(self, d, k, o):
    super(CP_CT3, self).__init__()
    self.layer_U1 = nn.ConvTranspose2d(d, k, kernel_size=4, stride=2, padding=1, bias=False)
    self.layer_U2 = nn.ConvTranspose2d(d, k, kernel_size=4, stride=2, padding=1, bias=False)
    self.layer_U3 = nn.ConvTranspose2d(d, k, kernel_size=3, stride=1, bias=False)
    self.layer_C = nn.ConvTranspose2d(k, o, kernel_size=4, stride=2, bias=True)
   #[-1, 1] 


  def forward(self, z):
     x1 = self.layer_U1(z)
     x2 = self.layer_U2(z) * x1 + x1 
     x3 = self.layer_U3(z) * x2 + x2 
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
        x2 = self.layer_U2(z) * x1 + x1 
        x3 = self.layer_U3(z) * x2 + x2 
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

class CP_L3_sparse_U(nn.Module):
    def __init__(self, d, k, o):
        super(CP_L3_sparse_U, self).__init__()
        
        self.layer_U1 = nn.Parameter(torch.triu(torch.randn(k, d)))     
        self.layer_U2 = nn.Parameter(torch.triu(torch.randn(k, d)))  
        self.layer_U3 = nn.Parameter(torch.triu(torch.randn(k, d)))  
        self.mask = torch.triu(torch.ones_like(self.layer_U1))
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

class CP_L3_sparse_L(nn.Module):
    def __init__(self, d, k, o):
        super(CP_L3_sparse_L, self).__init__()
        
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
        x_outer1 = torch.einsum('bi,bj->bij',(x1, x2))
        x_12 = torch.sum(x_outer1, 2)
        x_outer2 = torch.einsum('bi,bj->bij',(x_12, x3))
        x_123 = torch.sum(x_outer2, 2)
        x = self.layer_C(x_123) 
        return x