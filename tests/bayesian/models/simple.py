import torch

from pfs.ga.bayesian import Model, Variable, Proposal
from pfs.ga.bayesian.constants import Constants
from pfs.ga.bayesian.distributions import Normal
from pfs.ga.bayesian.proposals import NormalProposal

class Simple(Model):

    def __init__(self, N=Constants.MISSING, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N = N if N is not Constants.MISSING else (100,)

    def model(self, context):
        N = self.N

        theta = context.sample('theta', Normal(context.tensor(0.0), context.tensor(1.0)))

        with context.plate('n', N):
            x = context.sample('x', Normal(theta, context.tensor(1.0)))
            obs = context.sample('obs', Normal(x, context.tensor(0.5)), observed=True)

    def step(self, context):
        context.step(
            'theta',
            [ self.theta ],
            proposal = NormalProposal(
                self.theta.value(context.state),
                context.tensor(0.5))
        )

        context.step(
            'x',
            [ self.x ],
            proposal = NormalProposal(
                self.x.value(context.state),
                context.tensor(1.0)
            )
        )
