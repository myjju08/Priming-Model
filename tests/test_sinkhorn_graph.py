"""The execution optimization must preserve eager plans and stopping exactly."""
import os
import unittest
from unittest.mock import patch

import torch
from losses.discrepancy import featurewise_sinkhorn_plan


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA graph equivalence needs CUDA')
class SinkhornGraphTests(unittest.TestCase):
    def test_plans_stopping_and_envelope_gradients_are_bitwise_equal(self):
        for dtype in (torch.float32, torch.float64):
            for steps in (30, 100):
                for tolerance in (0., 1e-6, .1):
                    with self.subTest(dtype=dtype, steps=steps, tolerance=tolerance):
                        cost = torch.rand(3, 2, 7, 11, device='cuda', dtype=dtype)*.1
                        values = []
                        for flag in ('0', '1'):
                            with patch.dict(os.environ, PRIMING_CUDA_GRAPH_OT=flag):
                                values.append(featurewise_sinkhorn_plan(cost, epsilon=.01,
                                    num_iterations=steps, tolerance=tolerance))
                        for eager, optimized in zip(*values):
                            self.assertTrue(torch.equal(eager, optimized))
                        grads = []
                        for plan, _ in values:
                            variable = cost.clone().requires_grad_(True)
                            grads.append(torch.autograd.grad((plan*variable).sum(), variable)[0])
                        self.assertTrue(torch.equal(*grads))

    def test_nonfinite_cost_rejected(self):
        cost = torch.full((3, 2, 32, 32), float('nan'), device='cuda')
        with patch.dict(os.environ, PRIMING_CUDA_GRAPH_OT='1'), self.assertRaises(ValueError):
            featurewise_sinkhorn_plan(cost, epsilon=.01, num_iterations=100, tolerance=1e-6)


if __name__ == '__main__':
    unittest.main()
