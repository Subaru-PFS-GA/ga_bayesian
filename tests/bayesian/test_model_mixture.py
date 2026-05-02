import torch
import unittest

from pfs.ga.bayesian import Constants, model
from pfs.ga.bayesian import mcmc
from pfs.ga.bayesian.kernels import GibbsKernel
from pfs.ga.bayesian.mcmc import MCMC

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

        blanket_theta_1 = model.markov_blanket(model.theta_1)
        self.assertEqual(len(blanket_theta_1), 1)
        self.assertIn(model.x_1, blanket_theta_1)
        self.assertFalse(blanket_theta_1.has_selector)
        self.assertEqual(len(blanket_theta_1.selections), 0)

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

        blanket_x_1 = model.markov_blanket(model.x_1)
        self.assertEqual(len(blanket_x_1), 4)
        self.assertIn(model.theta_1, blanket_x_1)
        self.assertIn(model.x_2, blanket_x_1)
        self.assertIn(model.z, blanket_x_1)
        self.assertIn(model.obs, blanket_x_1)
        self.assertTrue(blanket_x_1.has_selector)
        self.assertIn(model.x, blanket_x_1.selections)

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

    def test_factor_graph(self):
        model = Mixture()
        model.build()

        factor_graph = model.factor_graph

        self.assertIsNotNone(factor_graph)
        self.assertEqual(len(factor_graph.sites), 7)
        self.assertEqual(len(factor_graph.factors), 7)

        factors_by_site_name = { factor.site.name: factor for factor in factor_graph.factors }

        # Root stochastic sites only depend on themselves.
        self.assertEqual(
            [site.name for site in factors_by_site_name["w"].scope],
            ["w"],
        )
        self.assertEqual(
            [site.name for site in factors_by_site_name["theta_1"].scope],
            ["theta_1"],
        )
        self.assertEqual(
            [site.name for site in factors_by_site_name["theta_2"].scope],
            ["theta_2"],
        )

        # z depends directly on w.
        self.assertEqual(
            [site.name for site in factors_by_site_name["z"].scope],
            ["w", "z"],
        )

        # x_1/x_2 depend on their corresponding theta variables and on z,
        # because downstream selection nodes gate their contribution.
        self.assertEqual(
            [site.name for site in factors_by_site_name["x_1"].scope],
            ["theta_1", "z", "x_1"],
        )
        self.assertEqual(
            [site.name for site in factors_by_site_name["x_2"].scope],
            ["theta_2", "z", "x_2"],
        )

        # obs depends on x (deterministic selection), so the factor scope includes
        # the stochastic ancestors of x: z, x_1, x_2, plus obs itself.
        self.assertEqual(
            [site.name for site in factors_by_site_name["obs"].scope],
            ["z", "x_1", "x_2", "obs"],
        )

    def test_site_log_prob(self):
        def site_log_prob_helper(N=(100,), C=()):
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

        site_log_prob_helper()
        site_log_prob_helper(C=(10,))
        
    def test_step_log_prob(self):
        def step_log_prob_helper(N=(100,), C=()):
            model = Mixture(N=N)
            model.build(batch_shape=C)
            state = model.sample()

            # Verify the shape of the log probabilities for each step
            self.assertEqual(model.steps['w'].log_prob(state).shape, C)
            
            self.assertEqual(model.steps['theta_1'].log_prob(state).shape, C)
            self.assertEqual(model.steps['theta_2'].log_prob(state).shape, C)

            self.assertEqual(model.steps['z'].log_prob(state).shape, N + C)

            self.assertEqual(model.steps['x_1'].log_prob(state).shape, N + C)
            self.assertEqual(model.steps['x_2'].log_prob(state).shape, N + C)

        step_log_prob_helper()
        step_log_prob_helper(C=(10,))

    def test_step_log_prob_gating(self):
        model = Mixture(N=(50,))
        model.build()

        state = model.sample()

        # Force all gates to select x_2. Under this configuration, the Gibbs
        # conditional for x_1 should not depend on x_1 values.
        gated_state = dict(state)
        gated_state[model.z.name] = torch.ones_like(model.z.value(state))

        state_a = dict(gated_state)
        state_b = dict(gated_state)

        state_a[model.x_1.name] = model.x_1.value(gated_state) + torch.randn_like(model.x_1.value(gated_state))
        state_b[model.x_1.name] = model.x_1.value(gated_state) + 3.0 * torch.randn_like(model.x_1.value(gated_state))

        lp_a = model.steps['x_1'].log_prob(state_a)
        lp_b = model.steps['x_1'].log_prob(state_b)

        torch.testing.assert_close(lp_a, lp_b)

    def test_run(self):
        def run_helper(N=(100,), C=1, samples=10):
            model = Mixture(N=N)
            model.build()

            # Generate some observed data
            init_state = model.sample()
            observed = { 'obs': init_state['obs'].clone() }

            kernel = GibbsKernel(model)
            mcmc = MCMC(kernel,
                        num_warmup=10, num_samples=samples, num_chains=C,
                        progress=False)
            
            mcmc.run(observed=observed)

            self.assertEqual(mcmc.trace['w'].shape, (samples, ) + (C, 2))
            
            self.assertEqual(mcmc.trace['theta_1'].shape, (samples, ) + (C,))
            self.assertEqual(mcmc.trace['theta_2'].shape, (samples, ) + (C,))

            self.assertEqual(mcmc.trace['z'].shape, (samples, ) + N + (C,))
            self.assertEqual(mcmc.trace['x_1'].shape, (samples, ) + N + (C,))
            self.assertEqual(mcmc.trace['x_2'].shape, (samples, ) + N + (C,))

            self.assertEqual(mcmc.trace['x'].shape, (samples, ) + N + (C,))

            self.assertEqual(mcmc.trace['obs'].shape, (samples, ) + N + (C,))

        # run_helper()
        run_helper(C=6)