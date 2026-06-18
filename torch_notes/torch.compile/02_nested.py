import torch

class InnerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(10, 20)
        self.lin2 = torch.nn.Linear(20, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.lin1(x))
        x = self.lin2(x)
        return x

class OuterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # compile the inner module
        self.inner_module = torch.compile(InnerModule())
        self.outer_lin = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = self.inner_module(x)
        return torch.nn.functional.relu(self.outer_lin(x))

t = torch.randn(5, 10)

outer_mod = OuterModule()
opt_outer_mod = torch.compile(outer_mod)

result = opt_outer_mod(t)
print(result.shape)