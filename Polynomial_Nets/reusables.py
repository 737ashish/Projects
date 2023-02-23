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
    
    

    
h1 = model.encoder(images[0])
z1, mu1, logvar1 = model.bottleneck(h1.to(device))
z1_1 = model.fc3(z1)
        #print('z.shape', z.shape)
r1 = model.decoder(z1_1)
h2 = model_s.encoder(images[0])
z2, mu2, logvar2 = model_s.bottleneck(h2.to(device))
z2_1 = model_s.fc3(z2)
        #print('z.shape', z.shape)
r = model_s.decoder(z2_1)
c = model_s.encoder.layer_c.to(torch.float64)
r, m, l = model(images)
z_s = images.reshape(-1, model_s.encoder.input_dimension)
T0_s = model_s.encoder.layer_a
T1_s = model_s.encoder.layer_b(z_s)
T2_s = (2 * torch.matmul(z_s, (model_s.encoder.mask * model_s.encoder.layer_c).T) * T1_s) - T0_s
T3_s = (2 * torch.matmul(z_s, (model_s.encoder.mask * model_s.encoder.layer_c).T) * T2_s) - T1_s
x_s = model_s.encoder.layer_C(T3_s)
mask = model_s.encoder.mask
z = images.reshape(-1, model.encoder.input_dimension)
T0 = model.encoder.layer_a
T1 = model.encoder.layer_b(z)
T2 = 2 * model.encoder.layer_c(z) * T1 - T0
T3 = 2 * model.encoder.layer_d(z) * T2 - T1
x = model.encoder.layer_C(T3)
sparse_int = torch.matmul(z_s, (model_s.encoder.mask * model_s.encoder.layer_c).T)
c_s = model_s.encoder.layer_c
c = model.encoder.layer_c.weight.to(torch.float64)