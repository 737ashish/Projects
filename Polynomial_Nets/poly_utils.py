import torch
import torch.nn as nn
import numpy as np

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
        model.epoch_end(epoch, loss, result)
        history.append(result)

    return history, losses

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

class Regression(nn.Module):
    def __init__(self, model, input_dim, loss_fn):
        super(Regression, self).__init__()
        
        self.model = model
        self.input_dimension = int(input_dim)
        self.loss_function = loss_fn

    def forward(self, z):
        x = self.model(z)
        return x
    
    def training_step(self, batch):
        input, target = batch[:, :self.input_dimension], batch[:, self.input_dimension:] 
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