"""
cuda graphs are usually captured for a fixed set of batch sizes. 
during runtime, requests are padded to the nearest captured shape and the corresponding graph is reused.

this mimics what inference engines do during decode, where the batch size changes continuously. 
instead of capturing a new graph for every request, the engine pre captures cuda graphs for a range of batch sizes and caches them. 
during inference, it selects the closest matching graph and pads the inputs if necessary.
"""
import torch
import torch.nn as nn

class PaddedGraphModel(nn.Module):
    SUPPORTED_BS = [1, 2, 4, 8, 16]

    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        ).cuda()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._capture()

    def _capture(self):
        """capture one graph per batch size."""
        for bs in self.SUPPORTED_BS:
            x = torch.randn(bs, 256, device="cuda")
            # warmup
            for _ in range(3):
                self.net(x)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            # static buffer for this batch size
            buf = torch.randn(bs, 256, device="cuda")
            with torch.cuda.graph(g):
                out = self.net(buf)
            
            # store them here
            self.graphs[bs] = (g, buf, out)

    def forward(self, x: torch.Tensor):
        """pad input to the nearest supported batch size and replay."""
        bs = x.size(0)
        # find nearest supported batch size (round up)
        padded_bs = min(b for b in self.SUPPORTED_BS if b >= bs)
        g, buf, out = self.graphs[padded_bs]

        # pad if needed
        if bs < padded_bs:
            x_padded = torch.cat([x, x[:padded_bs - bs]])
        else:
            x_padded = x

        buf.copy_(x_padded)
        g.replay()
        return out[:bs]


if __name__ == "__main__":
    model = PaddedGraphModel()
    # batch size changes between steps
    for bs in [1, 2, 4, 1, 8, 3, 16, 5]:
        x = torch.randn(bs, 256, device="cuda")
        y = model(x)
        print(f"bs={bs:>2d}  output shape={list(y.shape)}")
