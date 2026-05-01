import torch

from pfs.ga.bayesian import Model
from pfs.ga.bayesian.constants import Constants
from pfs.ga.bayesian.distributions import Normal, Uniform
from pfs.ga.bayesian.proposals import NormalProposal, MultivariateNormalProposal

class SinglePlate(Model):
    """
    This is a model where we define a variable inside a plate that
    has incoming edges from variables inside and outside the plate.
    """

    def __init__(self, N=Constants.MISSING):
        super().__init__()

        self.N = N if N is not Constants.MISSING else (100,)

    def model(self, context):
        N = self.N

        theta_1 = context.sample('theta_1', Normal(0.0, 1.0))
        theta_2 = context.sample('theta_2', Uniform(0.1, 1.0))

        with context.plate('n', N):
            x_1 = context.sample('x_1', Normal(theta_1, 1.0))
            x_2 = context.sample('x_2', Normal(x_1, theta_2))
            obs = context.sample('obs', Normal(x_2, 0.5), observed=True)

    def step(self, context):
        context.step(
            'theta',
            [ self.theta_1, self.theta_2 ],
            proposal = MultivariateNormalProposal(
                torch.stack([
                    self.theta_1.value(context.state),
                    self.theta_2.value(context.state)
                ], dim=-1),
                torch.tensor([ 1.0 ]) * torch.eye(2)
            )
        )
        
        context.step(
            'x',
            [ self.x_1, self.x_2 ],
            proposal = MultivariateNormalProposal(
                torch.stack([
                    self.x_1.value(context.state),
                    self.x_2.value(context.state)
                ], dim=-1),
                torch.tensor([ 1.0 ]) * torch.eye(2)
            )
        )
