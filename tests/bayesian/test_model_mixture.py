import torch
import unittest

from pfs.ga.bayesian import Constants
from pfs.ga.bayesian.kernels import GibbsKernel

from .models import Mixture

class TestModelMixture(unittest.TestCase):

    def test_build(self):
        model = Mixture(N=(100,))
        context = model._BuildContext(model)
        model.model(context)

        self.assertEqual([plate.name for plate in model.w.plates], [])
        self.assertEqual([parent.name for parent in model.w.parents], [])

        self.assertEqual([plate.name for plate in model.z.plates], ['n'])
        self.assertEqual([parent.name for parent in model.z.parents], ['w'])

        self.assertEqual([plate.name for plate in model.x_1.plates], ['n'])
        self.assertEqual([parent.name for parent in model.x_1.parents], ['theta_1'])

        self.assertEqual([plate.name for plate in model.x.plates], ['n'])
        self.assertEqual([parent.name for parent in model.x.parents], ['x_1', 'x_2', 'z'])

        # self.assertEqual([plate.name for plate in model.obs.plates], ['n'])
        self.assertEqual([parent.name for parent in model.obs.parents], ['x'])

        self.assertIn(model.z, model.w.children)
        self.assertIn(model.x_1, model.theta_1.children)
        self.assertIn(model.x, model.x_1.children)
        self.assertIn(model.x, model.x_2.children)
        self.assertIn(model.x, model.z.children)
        self.assertIn(model.obs, model.x.children)

    def test_block(self):
        def block_helper(N=(100,), C=()):
            model = Mixture(N=N)
            context = model._BuildContext(model)
            model.model(context)

            context = model._SampleContext(model, state={}, batch_shape=C)
            model.model(context)
            model.block(context, context.state)

            # Verify the Gibbs sampling steps are defined for the correct variables
            self.assertIn('w', model.steps)
            self.assertEqual(model.steps['w'].proposal.dist.event_shape, (2,))
            self.assertEqual(model.steps['w'].proposal.dist.batch_shape, C)

            self.assertIn('theta_1', model.steps)
            self.assertEqual(model.steps['theta_1'].proposal.dist.event_shape, ())
            self.assertEqual(model.steps['theta_1'].proposal.dist.batch_shape, C)

            self.assertIn('theta_2', model.steps)
            self.assertEqual(model.steps['theta_2'].proposal.dist.event_shape, ())
            self.assertEqual(model.steps['theta_2'].proposal.dist.batch_shape, C)

            self.assertIn('z', model.steps)
            self.assertEqual(model.steps['z'].proposal.dist.event_shape, ())
            self.assertEqual(model.steps['z'].proposal.dist.batch_shape, N + C)

            self.assertIn('x_1', model.steps)
            self.assertEqual(model.steps['x_1'].proposal.dist.event_shape, ())
            self.assertEqual(model.steps['x_1'].proposal.dist.batch_shape, N + C)

            self.assertIn('x_2', model.steps)
            self.assertEqual(model.steps['x_2'].proposal.dist.event_shape, ())
            self.assertEqual(model.steps['x_2'].proposal.dist.batch_shape, N + C)

        block_helper()
        block_helper(C=(10,))

    def test_sample(self):
        def sample_helper(N=(100,), C=()):
            model = Mixture(N=N)
            state = {}
            model.sample(state, batch_shape=C)

            self.assertEqual(model.w.shape(state), C + (2,))
            self.assertEqual(model.theta_1.shape(state), C)
            self.assertEqual(model.theta_2.shape(state), C)
            self.assertEqual(model.x_1.shape(state), N + C)
            self.assertEqual(model.x_2.shape(state), N + C)
            self.assertEqual(model.x.shape(state), N + C)
            self.assertEqual(model.obs.shape(state), N + C)

        sample_helper()
        sample_helper(C=(10,))

    def test_markov_blanket(self):
        model = Mixture()
        context = model._BuildContext(model)
        model.model(context)

        blanket_w = model.markov_blanket(model.w)
        self.assertEqual(len(blanket_w), 1)
        self.assertIn(model.z, blanket_w)

        blanket_theta_1 = model.markov_blanket(model.theta_1)
        self.assertEqual(len(blanket_theta_1), 4)
        self.assertIn(model.x_1, blanket_theta_1)
        self.assertIn(model.z, blanket_theta_1)
        self.assertIn(model.x, blanket_theta_1)
        self.assertIn(model.x_2, blanket_theta_1)
        self.assertNotIn(model.obs, blanket_theta_1)

        blanket_theta_2 = model.markov_blanket(model.theta_2)
        self.assertEqual(len(blanket_theta_2), 4)
        self.assertIn(model.x_2, blanket_theta_2)
        self.assertIn(model.z, blanket_theta_2)
        self.assertIn(model.x, blanket_theta_2)
        self.assertIn(model.x_1, blanket_theta_2)
        self.assertNotIn(model.obs, blanket_theta_2)

        blanket_z = model.markov_blanket(model.z)
        self.assertEqual(len(blanket_z), 5)
        self.assertIn(model.w, blanket_z)
        self.assertIn(model.x_1, blanket_z)
        self.assertIn(model.x_2, blanket_z)
        self.assertIn(model.x, blanket_z)
        self.assertIn(model.obs, blanket_z)

        blanket_x_1 = model.markov_blanket(model.x_1)
        self.assertEqual(len(blanket_x_1), 5)
        self.assertIn(model.theta_1, blanket_x_1)
        self.assertIn(model.x_2, blanket_x_1)
        self.assertIn(model.z, blanket_x_1)
        self.assertIn(model.x, blanket_x_1)
        self.assertIn(model.obs, blanket_x_1)

        blanket_x_2 = model.markov_blanket(model.x_2)
        self.assertEqual(len(blanket_x_2), 5)
        self.assertIn(model.theta_2, blanket_x_2)
        self.assertIn(model.x_1, blanket_x_2)
        self.assertIn(model.z, blanket_x_2)
        self.assertIn(model.x, blanket_x_2)
        self.assertIn(model.obs, blanket_x_2)

        blanket_x = model.markov_blanket(model.x)
        self.assertEqual(len(blanket_x), 4)
        self.assertIn(model.x_1, blanket_x)
        self.assertIn(model.x_2, blanket_x)
        self.assertIn(model.z, blanket_x)
        self.assertIn(model.obs, blanket_x)

        blanket_obs = model.markov_blanket(model.obs)
        self.assertEqual(len(blanket_obs), 4)
        self.assertIn(model.x_1, blanket_obs)
        self.assertIn(model.x_2, blanket_obs)
        self.assertIn(model.z, blanket_obs)
        self.assertIn(model.x, blanket_obs)

    # def test_log_prob(self):
    #     def log_prob_helper(N=(100,), C=()):
    #         model = Mixture(N=N, C=C)
    #         state = {}
    #         model.sample(state)

    #         lp = model.log_prob_theta_1(state)
    #         self.assertEqual(lp.shape, C)

    #         lp = model.log_prob_theta_2(state)
    #         self.assertEqual(lp.shape, C)

    #         lp = model.log_prob_z(state)
    #         self.assertEqual(lp.shape, N + C)

    #         lp = model.log_prob_x_1(state)
    #         self.assertEqual(lp.shape, N + C)

    #         lp = model.log_prob_x_2(state)
    #         self.assertEqual(lp.shape, N + C)

    #     log_prob_helper()
    #     log_prob_helper(C=(10,))
        

    # def test_step(self):
    #     def step_helper(N=(100,), C=()):
    #         model = Mixture(N=N, C=C)
    #         init_state = {}
    #         model.sample(init_state)
    #         model.build(init_state)

    #         kernel = GibbsKernel(model)
    #         final_state = kernel.step(init_state)
            
    #         self.assertEqual(final_state['theta_1'].shape, C)
    #         self.assertEqual(final_state['theta_2'].shape, C)
    #         self.assertEqual(final_state['z'].shape, N + C)
    #         self.assertEqual(final_state['x_1'].shape, N + C)
    #         self.assertEqual(final_state['x_2'].shape, N + C)
    #         self.assertEqual(final_state['x'].shape, N + C)
    #         self.assertEqual(final_state['obs'].shape, N + C)

    #     step_helper()
    #     step_helper(C=(10,))
