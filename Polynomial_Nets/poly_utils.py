import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from Cuda import DeviceDataLoader
import sys, inspect
from sklearn.preprocessing import MinMaxScaler

def MLP_model(in_dim, out_dim):
    model = nn.Sequential(
    nn.Linear(in_dim, 3 * in_dim),
    nn.ReLU(),
    nn.Linear(3 * in_dim, 2 * in_dim),
    nn.ReLU(),
    nn.Linear(2 * in_dim, in_dim),
    nn.ReLU(),
    nn.Linear(in_dim, out_dim)
    )
    return model

def training_regression(epochs, dimension, model, optimizer, device, loss_fn, train_loader):

    list_of_losses = []
    list_of_epochs = []

    for epoch in range(epochs):
        for batch in train_loader:   
            output = model(batch[:,:dimension].to(device))
            loss = loss_fn(output.to(device), batch[:,dimension:].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        list_of_losses.append(loss)
        list_of_epochs.append(epoch + 1)
        print('loss', loss)
        print("Epoch : ", epoch + 1)


def non_zero_count(model):
    num_param = []
    for name, param in model.named_parameters():
        num = torch.count_nonzero(param)
        num_param.append(num)

    count = torch.sum(torch.tensor(num_param))    
    return count

def sparse_model(model, threshold):
    state_dict = model.state_dict().copy()
    for name, p in model.named_parameters():
        mask1 = p > threshold
        mask2 = p < -threshold
        mask3 =  (mask1 | mask2) 
        p = nn.Parameter(mask3 * p)
        state_dict[name] = p

    model.load_state_dict(state_dict)
    return model

def prod_multi_index(mi1, mi2):
    sum_list = []
    for i in mi2:
        sum = mi1 + i
        sum_list.append(sum)
    sum_array = np.vstack(tuple(sum_list))
    mi3 = np.unique(np.array(sum_array), axis=0)
    return mi3

def add_multi_index(mi1, mi2):
    mi3 = np.unique(np.vstack((mi1, mi2)), axis=0)
    return mi3

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []
    losses = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        losses.append(loss)
        result = evaluate(model, val_loader)
        #model.epoch_end(epoch, loss, result)
        history.append(result)

    return history, losses

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

class Regression(nn.Module):
    def __init__(self, model, loss_fn):
        super(Regression, self).__init__()
        
        self.model = model
        self.input_dimension = model.input_dimension
        self.loss_function = loss_fn

    def forward(self, z):
        x = self.model(z)
        return x
    
    
    def training_step(self, batch):
        input, target = batch[:, :-1], batch[:, -1:] 
        out = self(input)                  # Generate predictions
        loss = self.loss_function(out, target) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        input, target = batch[:, :self.input_dimension], batch[:, self.input_dimension:] 
        out = self(input)                    # Generate predictions
        loss = self.loss_function(out, target)   # Calculate loss
        #acc = accuracy(out, labels)           # Calculate accuracy
        #return {'val_loss': loss, 'val_acc': acc}
        return {'val_loss': loss}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        #batch_accs = [x['val_acc'] for x in outputs]
        #epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        #return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, loss, result):
        print("Epoch [{}], train_loss: {:.4f} val_loss: {:.4f}".format(epoch, loss, result['val_loss']))

def params_CP(sparse_model, RANK, IN_DIM):
    #if isinstance(sparse_model, CP):
        sparse_param = []
        for name, p in sparse_model.named_parameters():  
            if len(p.shape) == 2:                
                if p.shape[0] == RANK and p.shape[1] == IN_DIM:
                    sparse_param = sparse_param + [p]
        return sparse_param

def multi_indices_list_CP(param_list):
    #if isinstance(model, CP):
        sparse_mi = []
        for i in param_list:
            ind = torch.where(i[0] == 0)[0].to('cpu').detach().numpy()
            mi_w = np.eye(i.shape[1])
            mi_w_s = np.delete(mi_w, ind, 0)
            sparse_mi = sparse_mi + [mi_w_s]
        return sparse_mi

def multi_indices_net_CP(sparse_mi):
        mi_s = sparse_mi[0]
        for i in range(1, len(sparse_mi)):
            mi1 = sparse_mi[i]
            mi_s = add_multi_index(prod_multi_index(mi_s, mi1), mi_s)
        return mi_s

def params_Cheby(model):
       sparse_param = []
       for name, p in model.named_parameters():            
            if name != 'layer_C.weight' and name != 'layer_C.bias' and name != 'T0':
                sparse_param = sparse_param + [p]
       return sparse_param
                
def multi_indices_list_Cheby(params):
        sparse_mi = []
        for i in params:
            ind = torch.where(i[0] == 0)[0].to('cpu').detach().numpy()
            mi_w = np.eye(i.shape[1])
            mi_w_s = np.delete(mi_w, ind, 0)
            sparse_mi = sparse_mi + [mi_w_s]
        return sparse_mi
        
def multi_indices_net_Cheby(sparse_mi, DEGREE):
    mi_list = []
    
    mi_s1 = sparse_mi[0]
    mi_s0 = np.zeros(mi_s1.shape[1])
    mi_list = mi_list + [mi_s1]
    for i in range(1, len(sparse_mi) + 1, 2):
        mi_s0 = add_multi_index(prod_multi_index(mi_s1, sparse_mi[i]), mi_s0)
        mi_list = mi_list + [mi_s0]
        
        if i == DEGREE - 1:
            mi_s = mi_s0
            return mi_s
        else:           
            mi_s1 = add_multi_index(prod_multi_index(mi_s0, sparse_mi[i+1]), mi_s1)
            mi_list = mi_list + [mi_s1]
            mi_s = mi_s1

    return mi_s

def generate_masks(degree, rank, in_dim):
    Masks = [torch.ones(rank, in_dim)]
    steps = []
    for i in range(0, degree - 1):
        M = torch.ones(rank, in_dim)
        r = torch.arange(i, rank, degree) 
        steps = steps + [r]
        M[torch.cat(steps, 0)] = 0
        Masks = Masks + [M]
    return Masks

def Norm_param(k, d):
    P = abs(nn.Linear(d, k, bias = False).weight.detach().to(torch.float32))
    P_sum = torch.sum(P, dim=1)
    P_norm = P/P_sum.reshape(-1,1)
    return P_norm


def Norm_param2(k, d):
    P = torch.ones(k,d)
    P_sum = torch.sum(P, dim=1)
    P_norm = P/P_sum.reshape(-1,1)
    return P_norm

def orthogonality_test_legendre(model1, model2, points, weights):
    return torch.dot((model1(points) * model2(points))[:,0], weights)

def orthogonality_test_chebyshev(model1, model2,  points, weights):
    return torch.dot((model1(points) * model2(points))[:,0] , weights)

def integral(model, points, weights):
    return torch.dot(model(points)[:,0], weights)

def orthogonal_matrix_legendre(models, points, weights):
    n = len(models)
    matrix = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            matrix[i,j] = orthogonality_test_legendre(models[i], models[j], points, weights)
    return matrix

def orthogonal_matrix_chebyshev(models, points, weights):
    n = len(models)
    matrix = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            matrix[i,j] = orthogonality_test_chebyshev(models[i], models[j], points, weights)
    return matrix

def round_matrix(matrix, threshold):
    mask1 = matrix > threshold
    mask2 = matrix < -threshold
    mask3 =  (mask1 | mask2) 
    matrix1 = mask3 * matrix
    return matrix1

def eval_at_one(models):
    one = torch.ones(1, models[0].input_dimension)
    return torch.tensor([i(one) for i in models])

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

def generate_data(N, IN_DIM, f):
    sample_domain = -1 + 2 * np.random.rand(N, IN_DIM) 
    input_tensor  = torch.tensor(sample_domain.reshape(-1,IN_DIM), dtype=torch.float32)
    target =  torch.tensor(f(sample_domain), dtype=torch.float32).reshape(-1,1)
    dataset = torch.cat((input_tensor, target), 1)
    #dataset = torch.tensor(torch.cat((torch.from_numpy(sample_domain).float(), torch.from_numpy(f).float())))
    return dataset

def generate_dataset_uqtf(test_function, N, target_scaler):
    xx_sample_dom_1 = -1 + 2 * np.random.rand(N, test_function.spatial_dimension)
    xx_sample = test_function.transform_sample(xx_sample_dom_1)
    yy_sample = test_function(xx_sample)
    target_scaler.fit(yy_sample.reshape(-1,1))
    yy_sample = target_scaler.transform(yy_sample.reshape(-1,1))
    input_tensor  = torch.tensor(xx_sample_dom_1, dtype=torch.float32)
    target =  torch.tensor(yy_sample, dtype=torch.float32)
    dataset = torch.cat((input_tensor, target), 1)
    return dataset, target_scaler

def data_split(dataset, train):
    N = len(dataset)
    N_t = int(N * train)
    N_v = N - N_t
    train_ds, val_ds = random_split(dataset, [N_t, N_v])
    return train_ds, val_ds

def eval_model(model, test_dataset):
    model = model.to('cpu').eval()
    outputs = model(test_dataset[:, :-1]).to('cpu').detach()
    return outputs

def create_loaders(N, in_dim, test_function, batch_size, device, split=0.9):
    dataset = generate_data(N, in_dim, test_function)
    train_ds, val_ds = data_split(dataset, split)
    train_loader = DeviceDataLoader(DataLoader(train_ds, batch_size, shuffle=True), device)
    val_loader = DeviceDataLoader(DataLoader(val_ds, batch_size), device)
    return train_loader, val_loader

def create_loaders_uqtf(N, test_function, batch_size, device, split=0.9, target_scaler=MinMaxScaler()):
    dataset, target_scaler = generate_dataset_uqtf(test_function, N, target_scaler)
    train_ds, val_ds = data_split(dataset, split)
    train_loader = DeviceDataLoader(DataLoader(train_ds, batch_size, shuffle=True), device)
    val_loader = DeviceDataLoader(DataLoader(val_ds, batch_size), device)
    return train_loader, val_loader

def train_model(NUM_EPOCHS, LEARNING_RATE, model, train_loader, val_loader, test_dataset, loss_fn, opt_func=torch.optim.Rprop, seed = 44):
    
    history, losses = fit(NUM_EPOCHS, LEARNING_RATE, model, train_loader, val_loader, opt_func)
    np.random.seed(seed)
    #test_dataset = ut.generate_data(N, IN_DIM, tf.runge2D)
    outputs = eval_model(model, test_dataset)
    test_target = test_dataset[:, -1:].to('cpu').detach()
    mse = loss_fn(outputs, test_target).detach().item()
    l_inf = torch.max(abs(outputs-test_target)).item()

    return model, history, losses, mse, l_inf

def train_model_uqtf(NUM_EPOCHS, LEARNING_RATE, model, train_loader, val_loader, test_dataset, target_scaler, loss_fn, opt_func=torch.optim.Rprop, seed = 44):
    
    history, losses = fit(NUM_EPOCHS, LEARNING_RATE, model, train_loader, val_loader, opt_func)
    np.random.seed(seed)
    #test_dataset = ut.generate_data(N, IN_DIM, tf.runge2D)
    outputs = eval_model(model, test_dataset)
    outputs = torch.tensor(target_scaler.inverse_transform(outputs))
    test_target = test_dataset[:, -1:].to('cpu').detach()
    test_target = torch.tensor(target_scaler.inverse_transform(test_target) )
    mse = loss_fn(outputs, test_target).detach().item()
    l_inf = torch.max(abs(outputs-test_target)).item()

    return model, history, losses, mse, l_inf

def get_models(module, arguments, device):
    classes = [cls_name for cls_name, cls_obj in inspect.getmembers(sys.modules[module]) if inspect.isclass(cls_obj)]
    return [getattr(sys.modules[module], i)(*arguments).to(device) for i in classes], classes

def get_functions(module):
    functions = [cls_name for cls_name, cls_obj in inspect.getmembers(sys.modules[module]) if inspect.isfunction(cls_obj)]
    return [getattr(sys.modules[module], i) for i in functions], functions