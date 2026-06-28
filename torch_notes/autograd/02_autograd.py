import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

grad = torch.autograd.grad(outputs=y, inputs=x)

print(grad)    # (tensor(4.),)
print(x.grad)  # None