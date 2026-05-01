import torch
import unittest

from pfs.ga.bayesian.kernels import GibbsKernel

from .models import SinglePlate

class TestModelSinglePlate(unittest.TestCase):

    def test_build(self):
        model = SinglePlate()
        context = model._BuildContext(model)
        model.model(context)

        self.assertIn(model.theta_1, model.x_1.parents)
        self.assertIn(model.x_1, model.theta_1.children)
        self.assertIn(model.x_2, model.x_1.children)

        self.assertEqual(len(model.theta_1.parents), 0)
        self.assertEqual(len(model.theta_1.children), 1)
        self.assertEqual(len(model.x_1.parents), 1)
        self.assertEqual(len(model.x_1.children), 1)
        self.assertEqual(len(model.x_2.parents), 2)
        self.assertEqual(len(model.x_2.children), 1)
        self.assertEqual(len(model.obs.parents), 1)
        self.assertEqual(len(model.obs.children), 0)
        
    def test_step(self):
        def step_helper(N=(100,), C=()):
            model = SinglePlate(N=N)
            context = model._BuildContext(model)
            model.model(context)

            context = model._SampleContext(model, state={}, batch_shape=C)
            model.model(context)
            model.step(context, context.state)

            # Verify the Gibbs sampling steps are defined for the correct variables
            self.assertIn('theta', model.steps)
            self.assertEqual(model.steps['theta'].proposal.dist.event_shape, (2,))
            self.assertEqual(model.steps['theta'].proposal.dist.batch_shape, C)

            self.assertIn('x', model.steps)
            self.assertEqual(model.steps['x'].proposal.dist.event_shape, (2,))
            self.assertEqual(model.steps['x'].proposal.dist.batch_shape, N + C)

        step_helper()
        step_helper(C=(10,))

    def test_sample(self):
        def sample_helper(N=(100,), C=()):
            model = SinglePlate(N=N)
            context = model._BuildContext(model)
            model.model(context)

            context = model._SampleContext(model, state={}, batch_shape=C)
            model.model(context)
            model.step(context, context.state)

            self.assertEqual(model.theta_1.shape(context.state), C)
            self.assertEqual(model.theta_2.shape(context.state), C)
            self.assertEqual(model.x_1.shape(context.state), N + C)
            self.assertEqual(model.x_2.shape(context.state), N + C)
            self.assertEqual(model.obs.shape(context.state), N + C)

        sample_helper()
        sample_helper(C=(10,))

    def test_log_prob(self):
        def log_prob_helper(N=(100,), C=()):
            model = SinglePlate(N=N)
            context = model._BuildContext(model)
            model.model(context)

            context = model._SampleContext(model, state={}, batch_shape=C)
            model.model(context)
            model.step(context, context.state)

            lp_theta_1 = model.theta_1.log_prob(context.state)
            lp_theta_2 = model.theta_2.log_prob(context.state)
            lp_x_1 = model.x_1.log_prob(context.state)
            lp_x_2 = model.x_2.log_prob(context.state)
            lp_obs = model.obs.log_prob(context.state)

            self.assertEqual(lp_theta_1.shape, C)
            self.assertEqual(lp_theta_2.shape, C)
            self.assertEqual(lp_x_1.shape, N + C)
            self.assertEqual(lp_x_2.shape, N + C)
            self.assertEqual(lp_obs.shape, N + C)

        # log_prob_helper()
        log_prob_helper(C=(10,))