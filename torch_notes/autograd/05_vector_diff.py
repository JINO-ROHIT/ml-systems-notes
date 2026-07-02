import torch

x = torch.tensor([1., 2.]).requires_grad_()
y = x + 1  # y = [2., 3.]

# backward() requires a scalar output, so explicitly sum y
y.sum().backward()
print("x.grad after y.sum().backward():", x.grad)  # [1., 1.]

# equivalently, pass a grad_output vector
x = torch.tensor([1., 2.]).requires_grad_()
y = x + 1
y.backward(gradient=torch.ones_like(y))
print("x.grad after y.backward(gradient=ones):", x.grad)  # [1., 1.]


