import torch
import unittest

from pfs.ga.bayesian import Constants
from pfs.ga.bayesian.kernels import GibbsKernel

from .models import Mixture

class TestModelMixture(unittest.TestCase):

    def test_build(self):
        model = Mixture()
        context = model._BuildContext(model)
        model.model(context)

        self.assertIn(model.w, model.z.parents)
        self.assertIn(model.theta_1, model.x_1.parents)
        self.assertIn(model.theta_2, model.x_2.parents)
        self.assertIn(model.z, model.x.parents)
        self.assertIn(model.x_1, model.x.parents)
        self.assertIn(model.x_2, model.x.parents)
        self.assertIn(model.x, model.obs.parents)

    def test_step(self):
        def step_helper(N=(100,), C=()):
            model = Mixture(N=N)
            model.build(batch_shape=C)

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

        step_helper()
        step_helper(C=(10,))

    def test_sample(self):
        def sample_helper(N=(100,), C=()):
            model = Mixture(N=N)
            model.build(batch_shape=C)
            state = model.sample()

            self.assertEqual(model.w.shape(state), C + (2,))
            self.assertEqual(model.z.shape(state), N + C)
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
        model.build()

        blanket_w = model.markov_blanket(model.w)
        self.assertEqual(len(blanket_w), 1)
        self.assertIn(model.z, blanket_w)
        self.assertFalse(blanket_w.has_selector)
        self.assertEqual(len(blanket_w.selections), 0)
        self.assertEqual(len(blanket_w.edges), 1)
        self.assertEqual(blanket_w.edges[0].source, model.w)
        self.assertEqual(blanket_w.edges[0].target, model.z)
        self.assertEqual(blanket_w.edges[0].role, 'child')

        blanket_theta_1 = model.markov_blanket(model.theta_1)
        self.assertEqual(len(blanket_theta_1), 1)
        self.assertIn(model.x_1, blanket_theta_1)
        self.assertFalse(blanket_theta_1.has_selector)
        self.assertEqual(len(blanket_theta_1.selections), 0)
        self.assertEqual(len(blanket_theta_1.edges), 1)
        self.assertEqual(blanket_theta_1.edges[0].source, model.theta_1)
        self.assertEqual(blanket_theta_1.edges[0].target, model.x_1)
        self.assertEqual(blanket_theta_1.edges[0].role, 'child')

        blanket_theta_2 = model.markov_blanket(model.theta_2)
        self.assertEqual(len(blanket_theta_2), 1)
        self.assertIn(model.x_2, blanket_theta_2)
        self.assertFalse(blanket_theta_2.has_selector)
        self.assertEqual(len(blanket_theta_2.selections), 0)

        blanket_z = model.markov_blanket(model.z)
        self.assertEqual(len(blanket_z), 4)
        self.assertIn(model.w, blanket_z)
        self.assertIn(model.x_1, blanket_z)
        self.assertIn(model.x_2, blanket_z)
        self.assertIn(model.obs, blanket_z)
        self.assertTrue(blanket_z.has_selector)
        self.assertIn(model.x, blanket_z.selections)

        edge_z_parent = [edge for edge in blanket_z.edges if edge.source is model.w and edge.target is model.z and edge.role == 'parent']
        self.assertEqual(len(edge_z_parent), 1)
        self.assertEqual(len(edge_z_parent[0].selections), 0)

        edge_z_child = [edge for edge in blanket_z.edges if edge.source is model.z and edge.target is model.obs and edge.role == 'child']
        self.assertEqual(len(edge_z_child), 1)
        self.assertIn(model.x, edge_z_child[0].selections)

        edge_z_coparent_1 = [edge for edge in blanket_z.edges if edge.source is model.x_1 and edge.target is model.obs and edge.role == 'coparent']
        edge_z_coparent_2 = [edge for edge in blanket_z.edges if edge.source is model.x_2 and edge.target is model.obs and edge.role == 'coparent']
        self.assertEqual(len(edge_z_coparent_1), 1)
        self.assertEqual(len(edge_z_coparent_2), 1)
        self.assertIn(model.x, edge_z_coparent_1[0].selections)
        self.assertIn(model.x, edge_z_coparent_2[0].selections)

        blanket_x_1 = model.markov_blanket(model.x_1)
        self.assertEqual(len(blanket_x_1), 4)
        self.assertIn(model.theta_1, blanket_x_1)
        self.assertIn(model.x_2, blanket_x_1)
        self.assertIn(model.z, blanket_x_1)
        self.assertIn(model.obs, blanket_x_1)
        self.assertTrue(blanket_x_1.has_selector)
        self.assertIn(model.x, blanket_x_1.selections)
        self.assertEqual(len(blanket_x_1.edges), 4)

        blanket_x_2 = model.markov_blanket(model.x_2)
        self.assertEqual(len(blanket_x_2), 4)
        self.assertIn(model.theta_2, blanket_x_2)
        self.assertIn(model.x_1, blanket_x_2)
        self.assertIn(model.z, blanket_x_2)
        self.assertIn(model.obs, blanket_x_2)
        self.assertTrue(blanket_x_2.has_selector)
        self.assertIn(model.x, blanket_x_2.selections)

        blanket_x = model.markov_blanket(model.x)
        self.assertEqual(len(blanket_x), 4)
        self.assertIn(model.x_1, blanket_x)
        self.assertIn(model.x_2, blanket_x)
        self.assertIn(model.z, blanket_x)
        self.assertIn(model.obs, blanket_x)
        self.assertTrue(blanket_x.has_selector)
        self.assertIn(model.x, blanket_x.selections)

        blanket_obs = model.markov_blanket(model.obs)
        self.assertEqual(len(blanket_obs), 3)
        self.assertIn(model.x_1, blanket_obs)
        self.assertIn(model.x_2, blanket_obs)
        self.assertIn(model.z, blanket_obs)
        self.assertTrue(blanket_obs.has_selector)
        self.assertIn(model.x, blanket_obs.selections)

        edge_obs_parent_z = [edge for edge in blanket_obs.edges if edge.source is model.z and edge.target is model.obs and edge.role == 'parent']
        edge_obs_parent_x1 = [edge for edge in blanket_obs.edges if edge.source is model.x_1 and edge.target is model.obs and edge.role == 'parent']
        edge_obs_parent_x2 = [edge for edge in blanket_obs.edges if edge.source is model.x_2 and edge.target is model.obs and edge.role == 'parent']
        self.assertEqual(len(edge_obs_parent_z), 1)
        self.assertEqual(len(edge_obs_parent_x1), 1)
        self.assertEqual(len(edge_obs_parent_x2), 1)
        self.assertIn(model.x, edge_obs_parent_z[0].selections)
        self.assertIn(model.x, edge_obs_parent_x1[0].selections)
        self.assertIn(model.x, edge_obs_parent_x2[0].selections)

    def test_log_prob(self):
        def log_prob_helper(N=(100,), C=()):
            model = Mixture(N=N)
            model.build(batch_shape=C)
            state = model.sample()

            lp = model.w.log_prob(state)
            self.assertEqual(lp.shape, C)

            lp = model.theta_1.log_prob(state)
            self.assertEqual(lp.shape, C)

            lp = model.theta_2.log_prob(state)
            self.assertEqual(lp.shape, C)

            lp = model.z.log_prob(state)
            self.assertEqual(lp.shape, N + C)

            lp = model.x_1.log_prob(state)
            self.assertEqual(lp.shape, N + C)

            lp = model.x_2.log_prob(state)
            self.assertEqual(lp.shape, N + C)

            lp = model.obs.log_prob(state)
            self.assertEqual(lp.shape, N + C)

        log_prob_helper()
        log_prob_helper(C=(10,))
        
