import torch
import unittest

from pfs.ga.bayesian.kernels import GibbsKernel

from .models import Simple

class TestModelSimple(unittest.TestCase):

    def test_sample(self):
        model = Simple()
        state = {}
        model.sample(state)

        self.assertIn('theta', state)
        self.assertIn('x', state)

    def test_log_prob(self):
        model = Simple()
        state = {}
        model.sample(state)

        lp = model.log_prob_x_given_all(state)
        self.assertIsInstance(lp, torch.Tensor)

    def test_build(self):
        model = Simple()
        init_state = {}
        model.sample(init_state)

        model.build(init_state)

        self.assertIn('x', model.proposals)
        self.assertIn('x', model.steps)

    def test_step(self):
        model = Simple()
        init_state = {}
        model.sample(init_state)
        model.build(init_state)

        kernel = GibbsKernel(model)
        final_state = kernel.step(init_state)
        
        self.assertIn('theta', final_state)
        self.assertIn('x', final_state)