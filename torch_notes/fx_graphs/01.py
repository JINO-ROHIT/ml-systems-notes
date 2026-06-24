
# import torch
# import torch.fx

# class MyModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.param = torch.nn.Parameter(torch.rand(3, 4))
#         self.linear = torch.nn.Linear(4, 5)

#     def forward(self, x):
#         return torch.topk(torch.sum(
#             self.linear(x + self.linear.weight).relu(), dim=-1), 3)

# m = MyModule()
# gm = torch.fx.symbolic_trace(m)

# print(gm.graph)

# gm.graph.print_tabular()

# print(gm.code)


# from torch.fx import symbolic_trace
# from torch.fx.passes.graph_drawer import FxGraphDrawer

# gm = symbolic_trace(m)

# drawer = FxGraphDrawer(gm, "my_graph")
# dot = drawer.get_dot_graph()

# dot.write_svg("01.svg")



import torch
import torch.fx

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)

    def forward(self, x):
        return torch.relu(self.linear(x))

gm = torch.fx.symbolic_trace(MyModel())
gm.graph.print_tabular()

print(gm.graph)