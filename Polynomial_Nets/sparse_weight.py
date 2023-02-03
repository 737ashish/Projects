import torch
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.tril(torch.randn(hidden_size, input_size)))
        self.bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        x = x @ self.weight.t() + self.bias
        return x

# Initialize model
model = MyModel(input_size=10, hidden_size=5)

# Freeze the upper triangular part of the weight matrix
mask = torch.triu(torch.ones(model.hidden_size, model.input_size))
model.weight.data = model.weight.data * mask
model.weight.data[mask == 0] = torch.nn.functional.detach(model.weight.data[mask == 0])


#part2
param_vector = torch.nn.utils.parameters_to_vector(model.parameters())
param_vector.grad = torch.zeros_like(param_vector)
torch.nn.utils.vector_to_parameters(param_vector, model.parameters())

for param in model.parameters():
    param.grad = None

for param in model.layer1.parameters():
    if param.requires_grad:
        param.grad = None


optimizer.zero_grad()
optimizer.step()
