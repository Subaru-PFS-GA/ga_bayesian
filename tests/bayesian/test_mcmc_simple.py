import torch
import unittest

from pfs.ga.bayesian import MCMC
from pfs.ga.bayesian.kernels import GibbsKernel

from .models import Simple

class TestMCMCSimple(unittest.TestCase):

    pass

    def test_step(self):
        def run_helper(N=(100,), C=1, samples=10):
            model = Simple(N=N)
            model.build()

            # Generate some observed data
            init_state = model.sample()
            observed = { 'obs': init_state['obs'].clone() }

            kernel = GibbsKernel(model)
            mcmc = MCMC(kernel,
                        num_warmup=10, num_samples=samples, num_chains=C,
                        progress=False)
            
            mcmc.run(observed=observed)

            self.assertEqual(mcmc.trace['theta'].shape, (samples, ) + (C,))
            self.assertEqual(mcmc.trace['x'].shape, (samples, ) + N + (C,))
            self.assertEqual(mcmc.trace['obs'].shape, (samples, ) + N + (C,))

        # run_helper()
        run_helper(C=6)