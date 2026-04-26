import torch
import unittest

from pfs.ga.bayesian import Constants
from pfs.ga.bayesian.kernels import GibbsKernel

from .models import Mixture

class TestModelMixture(unittest.TestCase):

    def test_sample(self):
        def sample_helper(N=(100,), C=()):
            model = Mixture(N=N, C=C)
            state = {}
            model.sample(state)

            self.assertEqual(state['theta_1'].shape, C)
            self.assertEqual(state['theta_2'].shape, C)
            self.assertEqual(state['x_1'].shape, N + C)
            self.assertEqual(state['x_2'].shape, N + C)
            self.assertEqual(state['x'].shape, N + C)
            self.assertEqual(state['obs'].shape, N + C)

        sample_helper()
        sample_helper(C=(10,))

    def test_log_prob(self):
        def log_prob_helper(N=(100,), C=()):
            model = Mixture(N=N, C=C)
            state = {}
            model.sample(state)

            lp = model.log_prob_theta_1(state)
            self.assertEqual(lp.shape, C)

            lp = model.log_prob_theta_2(state)
            self.assertEqual(lp.shape, C)

            lp = model.log_prob_z(state)
            self.assertEqual(lp.shape, N + C)

            lp = model.log_prob_x_1(state)
            self.assertEqual(lp.shape, N + C)

            lp = model.log_prob_x_2(state)
            self.assertEqual(lp.shape, N + C)

        log_prob_helper()
        log_prob_helper(C=(10,))
        

    def test_build(self):
        def build_helper(N=(100,), C=()):
            model = Mixture(N=N, C=C)
            init_state = {}
            model.sample(init_state)

            model.build(init_state)

            self.assertEqual(model.proposal_theta_1.batch_shape, C)
            self.assertEqual(model.proposal_theta_1.event_shape, ())

            self.assertEqual(model.proposal_theta_2.batch_shape, C)
            self.assertEqual(model.proposal_theta_2.event_shape, ())

            self.assertEqual(model.proposal_z.batch_shape, N + C)
            self.assertEqual(model.proposal_z.event_shape, ())

            self.assertEqual(model.proposal_x_1.batch_shape, N + C)
            self.assertEqual(model.proposal_x_1.event_shape, ())

            self.assertEqual(model.proposal_x_2.batch_shape, N + C)
            self.assertEqual(model.proposal_x_2.event_shape, ())        
            
            self.assertIn('theta_1', model.steps)
            self.assertIn('theta_2', model.steps)
            self.assertIn('z', model.steps)
            self.assertIn('x_1', model.steps)
            self.assertIn('x_2', model.steps)

        build_helper()
        build_helper(C=(10,))

    def test_step(self):
        def step_helper(N=(100,), C=()):
            model = Mixture(N=N, C=C)
            init_state = {}
            model.sample(init_state)
            model.build(init_state)

            kernel = GibbsKernel(model)
            final_state = kernel.step(init_state)
            
            self.assertEqual(final_state['theta_1'].shape, C)
            self.assertEqual(final_state['theta_2'].shape, C)
            self.assertEqual(final_state['z'].shape, N + C)
            self.assertEqual(final_state['x_1'].shape, N + C)
            self.assertEqual(final_state['x_2'].shape, N + C)
            self.assertEqual(final_state['x'].shape, N + C)
            self.assertEqual(final_state['obs'].shape, N + C)

        step_helper()
        step_helper(C=(10,))