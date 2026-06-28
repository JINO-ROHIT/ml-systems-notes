import torch

x = torch.tensor(2.).requires_grad_()
y = torch.tensor(3.).requires_grad_()

z = x * x * y

# grad_x = torch.autograd.grad(outputs=z, inputs=x) # this mean dz/dx
# grad_y = torch.autograd.grad(outputs=z, inputs=y)
# print(grad_x[0], grad_y[0])


## the problem is after the first forward pass, the graph is released, so you need to explicitly retain graph. 
## this is also the same for backward as well.


grad_x = torch.autograd.grad(outputs=z, inputs=x, retain_graph=True) # this mean dz/dx
grad_y = torch.autograd.grad(outputs=z, inputs=y)
print(grad_x[0], grad_y[0])