import torch
import unittest

from pfs.ga.bayesian import Constants
from pfs.ga.bayesian.kernels import GibbsKernel

from .models import JointProposal

class TestModelJointProposal(unittest.TestCase):

    def test_build(self):
        model = JointProposal(N=(100,))
        context = model._BuildContext(model)
        model.model(context)

        self.assertEqual([plate.name for plate in model.mu.plates], [])
        self.assertEqual([parent.name for parent in model.mu.parents], [])

        self.assertEqual([plate.name for plate in model.sigma.plates], [])
        self.assertEqual([parent.name for parent in model.sigma.parents], [])

        self.assertEqual([plate.name for plate in model.x.plates], ['n'])
        self.assertEqual([parent.name for parent in model.x.parents], ['mu', 'sigma'])

        self.assertEqual([plate.name for plate in model.obs.plates], ['n'])
        self.assertEqual([parent.name for parent in model.obs.parents], ['x'])

        self.assertIn(model.x, model.mu.children)
        self.assertIn(model.x, model.sigma.children)
        self.assertIn(model.obs, model.x.children)

    def test_block(self):
        def block_helper(N=(100,), C=()):
            model = JointProposal(N=N)
            context = model._BuildContext(model)
            model.model(context)

            context = model._SampleContext(model, state={}, batch_shape=C)
            model.model(context)
            model.block(context, context.state)

            self.assertIn('theta', model.steps)
            self.assertEqual(model.steps['theta'].sites, [model.mu, model.sigma])
            self.assertEqual(model.steps['theta'].proposal.dist.event_shape, (2,))
            self.assertEqual(model.steps['theta'].proposal.dist.batch_shape, C)

            self.assertIn('x', model.steps)
            self.assertEqual(model.steps['x'].sites, [model.x])
            self.assertEqual(model.steps['x'].proposal.dist.event_shape, ())
            self.assertEqual(model.steps['x'].proposal.dist.batch_shape, N + C)

        block_helper()
        block_helper(C=(10,))

    def test_sample(self):
        def sample_helper(N=(100,), C=()):
            model = JointProposal(N=N)
            state = {}
            model.sample(state, batch_shape=C)

            self.assertEqual(model.mu.shape(state), C)
            self.assertEqual(model.sigma.shape(state), C)
            self.assertEqual(model.x.shape(state), N + C)
            self.assertEqual(model.obs.shape(state), N + C)

        sample_helper()
        sample_helper(C=(10,))

    def test_markov_blanket(self):
        model = JointProposal()
        context = model._BuildContext(model)
        model.model(context)

        blanket_mu = model.markov_blanket(model.mu)
        self.assertEqual(len(blanket_mu), 2)
        self.assertIn(model.sigma, blanket_mu)
        self.assertIn(model.x, blanket_mu)

        blanket_sigma = model.markov_blanket(model.sigma)
        self.assertEqual(len(blanket_sigma), 2)
        self.assertIn(model.mu, blanket_sigma)
        self.assertIn(model.x, blanket_sigma)

        blanket_theta = model.markov_blanket([ model.mu, model.sigma ])
        self.assertEqual(len(blanket_theta), 1)
        self.assertIn(model.x, blanket_theta)

        blanket_x = model.markov_blanket(model.x)
        self.assertEqual(len(blanket_x), 3)
        self.assertIn(model.mu, blanket_x)
        self.assertIn(model.sigma, blanket_x)
        self.assertIn(model.obs, blanket_x)

        blanket_obs = model.markov_blanket(model.obs)
        self.assertEqual(len(blanket_obs), 1)
        self.assertIn(model.x, blanket_obs)

