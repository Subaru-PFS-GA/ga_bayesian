import torch

from pfs.ga.bayesian import Model, Variable, Proposal
from pfs.ga.bayesian.constants import Constants
from pfs.ga.bayesian.distributions import Normal
from pfs.ga.bayesian.proposals import NormalProposal

class Simple(Model):

    def __init__(self, N=Constants.MISSING, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N = N = N if N is not Constants.MISSING else (100,)

    def model(self, context):
        N = self.N

        theta = context.sample('theta', Normal(0.0, 1.0))

        with context.plate('n', N):
            x = context.sample('x', Normal(theta, 1.0))
            obs = context.sample('obs', Normal(x, 0.5), observed=True)

    def step(self, context, state):
        context.step('theta', [ self.theta ], proposal=NormalProposal(self.theta.value(state), 0.5))
        context.step('x', [ self.x ], proposal=NormalProposal(self.x.value(state), 1.0))
