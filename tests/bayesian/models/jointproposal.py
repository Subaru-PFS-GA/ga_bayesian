import torch

from pfs.ga.bayesian import Constants
from pfs.ga.bayesian import Model
from pfs.ga.bayesian.distributions import Normal, Uniform
from pfs.ga.bayesian.proposals import MultivariateNormalProposal, NormalProposal


class JointProposal(Model):

    def __init__(self, N=Constants.MISSING, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N = N if N is not Constants.MISSING else (100,)

    def model(self, context):
        N = self.N

        # Population-level parameters (outside plate)
        mu = context.sample('mu', Normal(0.0, 5.0, validate_args=False))
        sigma = context.sample('sigma', Uniform(0.1, 3.0, validate_args=False))

        # Member-level latent variable and observation (inside plate)
        with context.plate('n', N):
            x = context.sample('x', Normal(mu, sigma, validate_args=False))
            obs = context.sample('obs', Normal(x, 0.25, validate_args=False), observed=True)

    def step(self, context):
        # Joint step for population parameters [mu, sigma]
        context.step(
            'theta',
            [ self.mu, self.sigma ],
            proposal = MultivariateNormalProposal(
                torch.zeros(self.mu.shape(context.state) + (2,)),
                (torch.eye(2) * 0.5).expand(self.mu.shape(context.state) + (2, 2))
            )
        )

        # Member-level latent variable step
        context.step(
            'x',
            [ self.x ],
            proposal = NormalProposal(
                torch.zeros(self.x.shape(context.state)),
                torch.ones(self.x.shape(context.state)) * 0.5
            )
        )
