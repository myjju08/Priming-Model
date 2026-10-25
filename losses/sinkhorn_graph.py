"""Replay the unchanged eager Sinkhorn blocks as a CUDA graph.

Device convergence checks remain at exactly the original ten-update boundaries.
The host polls every 100 iterations; inactive features retain their exact duals.
No compiler, changed reductions, relaxed tolerance, or approximate plans are used.
"""
import math
import torch

_CACHE = {}


class SinkhornGraph:
    def __init__(self, kernel, num_iterations, tolerance):
        self.kernel = torch.empty_like(kernel)
        features, _, rows, columns = kernel.shape
        self.u = torch.zeros_like(kernel[..., :, 0])
        self.v = torch.zeros_like(kernel[..., 0, :])
        self.active = torch.ones(features, dtype=torch.bool, device=kernel.device)
        self.iterations = torch.full((features,), num_iterations, dtype=torch.int64, device=kernel.device)
        self.counter = torch.zeros((), dtype=torch.int64, device=kernel.device)
        self.block_size = 100 if num_iterations % 100 == 0 else 10

        def block():
            for _ in range(self.block_size//10):
                for _ in range(10):
                    new_u = -math.log(rows)-torch.logsumexp(self.kernel+self.v.unsqueeze(-2), -1)
                    new_v = -math.log(columns)-torch.logsumexp(self.kernel+new_u.unsqueeze(-1), -2)
                    self.u.copy_(torch.where(self.active[:, None, None], new_u, self.u))
                    self.v.copy_(torch.where(self.active[:, None, None], new_v, self.v))
                self.counter.add_(10)
                if tolerance > 0:
                    plan = torch.exp(self.u.unsqueeze(-1)+self.kernel+self.v.unsqueeze(-2))
                    row_error = (plan.sum(-1)-1./rows).abs().amax(dim=(1, 2))
                    column_error = (plan.sum(-2)-1./columns).abs().amax(dim=(1, 2))
                    converged = self.active & (torch.maximum(row_error, column_error) <= tolerance)
                    self.iterations.copy_(torch.where(converged, self.counter, self.iterations))
                    self.active.logical_and_(~converged)

        self.reset(kernel, num_iterations)
        stream = torch.cuda.Stream(device=kernel.device)
        stream.wait_stream(torch.cuda.current_stream(kernel.device))
        with torch.cuda.stream(stream):
            block()
        torch.cuda.current_stream(kernel.device).wait_stream(stream)
        self.reset(kernel, num_iterations)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            block()

    def reset(self, kernel, num_iterations):
        self.kernel.copy_(kernel)
        self.u.zero_()
        self.v.zero_()
        self.active.fill_(True)
        self.iterations.fill_(num_iterations)
        self.counter.zero_()

    def solve(self, kernel, num_iterations, tolerance):
        self.reset(kernel, num_iterations)
        for _ in range(num_iterations//self.block_size):
            self.graph.replay()
            if tolerance > 0 and not bool(self.active.any()):
                break
        plan = torch.exp(self.u.unsqueeze(-1)+self.kernel+self.v.unsqueeze(-2))
        return plan, self.iterations.clone()


def solve(kernel, *, num_iterations, tolerance):
    key = (kernel.device, tuple(kernel.shape), kernel.dtype, num_iterations, tolerance)
    with torch.cuda.device(kernel.device):
        if key not in _CACHE:
            _CACHE[key] = SinkhornGraph(kernel, num_iterations, tolerance)
        return _CACHE[key].solve(kernel, num_iterations, tolerance)
