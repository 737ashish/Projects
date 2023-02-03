optimizer.add_param_group({'params': filter_params(model.named_parameters()), 'lr': 0.01})
out_features, in_features = model.decoder[0].layer_U1.weight.shape
weight = torch.tril(torch.randn(out_features, in_features))
before_train = weight
model.decoder[0].layer_U1.weight = nn.Parameter(weight)
#weight = model.decoder[0].layer_U1.weight
triu_indices = torch.tril_indices(out_features, in_features)
triu = model.decoder[0].layer_U1.weight.data[triu_indices]
triu1 = weight[triu_indices[0,1]]

non_zero_ind = torch.nonzero(model.decoder[0].layer_U1.weight, as_tuple=True)


#mask = torch.tril(torch.ones(out_features, in_features))
#mult = model.decoder[0].layer_U1.weight.data * mask
#model.decoder[0].layer_U1.weight.data = model.decoder[0].layer_U1.weight.data * mask
#model.decoder[0].layer_U1.weight.data[mask == 0].requires_grad = False
#model.decoder[0].layer_U1.weight.data[mask == 0] = model.decoder[0].layer_U1.weight.data[mask == 0].detach()
elements = [model.decoder[0].layer_U1.weight[coord] for coord in zip(*triu_indices)]
non_zero = [model.decoder[0].layer_U1.weight[coord] for coord in zip(*non_zero_ind)]
param_list = []
param_list_name =[]
for name, param in model.named_parameters():
    if name == 'decoder.0.layer_U1.weight':
        continue
    
    param_list.append(param)
    param_list_name.append(name)

#elements = torch.Tensor(elements)
optimizer = torch.optim.Adam(param_list + elements, lr=LEARNING_RATE)
for p in model.named_parameters():
    print(p.shape)

params = model.parameters()[0]
model.decoder[0].layer_U1.weight.data

for name, p in model.named_parameters():
    print(name)
    if 'decoder.0.layer_U1.weight' in name and mask[p == 0].all():
        p.grad = None 
    
    
    
