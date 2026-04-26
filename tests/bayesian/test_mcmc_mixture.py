import torch
import unittest

from pfs.ga.bayesian import MCMC
from pfs.ga.bayesian.kernels import GibbsKernel

from .models import Mixture

class TestMCMCMixture(unittest.TestCase):

    def test_step(self):
        def run_helper(N=(100,), C=(), samples=10):
            model = Mixture(N=N, C=C)
            init_state = {}
            model.sample(init_state)
            model.build(init_state)

            if len(C) == 0:
                observed = {
                    'obs': init_state['obs'].clone()
                }
            else:
                # Copy the first chain to all chains
                observed = {
                    'obs': init_state['obs'][..., :1].tile(init_state['obs'].shape[-1]),
                }

            kernel = GibbsKernel(model)
            mcmc = MCMC(kernel, num_warmup=10, num_samples=samples, progress=False)
            
            mcmc.run(observed=observed)

            self.assertEqual(mcmc.trace['w'].shape, (samples, ) + C + (2,))
            self.assertEqual(mcmc.trace['z'].shape, (samples, ) + N + C)
            self.assertEqual(mcmc.trace['theta_1'].shape, (samples, ) + C)
            self.assertEqual(mcmc.trace['theta_2'].shape, (samples, ) + C)
            self.assertEqual(mcmc.trace['x_1'].shape, (samples, ) + N + C)
            self.assertEqual(mcmc.trace['x_2'].shape, (samples, ) + N + C)
            self.assertEqual(mcmc.trace['x'].shape, (samples, ) + N + C)
            self.assertEqual(mcmc.trace['obs'].shape, (samples, ) + N + C)

        run_helper()
        run_helper(C=(10,))