import torch

from pfs.ga.bayesian import Constants
from pfs.ga.bayesian import Model
from pfs.ga.bayesian import torch_extensions
from pfs.ga.bayesian.distributions import Uniform, Dirichlet, Categorical, Normal
from pfs.ga.bayesian.proposals import DirichletProposal, CategoricalProposal, MultivariateNormalProposal, NormalProposal

class Mixture(Model):

    def __init__(self, N=Constants.MISSING, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N = N = N if N is not Constants.MISSING else (100,)

    def model(self, context):
        K = 2
        N = self.N

        w = context.sample('w', Dirichlet(5.0 * torch.ones(K)))
        theta_1 = context.sample('theta_1', Uniform(-1.0, 1.0, validate_args=False))
        theta_2 = context.sample('theta_2', Uniform(2.0, 3.0, validate_args=False))

        with context.plate("n", N):
            z = context.sample('z', Categorical(w))
            x_1 = context.sample('x_1', Normal(theta_1, 1.0))
            x_2 = context.sample('x_2', Normal(theta_2, 3.0))
            x = context.select('x', [x_1, x_2], z)
            obs = context.sample('obs', Normal(x, 0.25), observed=True)

    def block(self, context, state):
        K = 2

        # Define the Gibbs sampling steps for each group of sampled variables
        context.step('w', [ self.w ], proposal=DirichletProposal(3.0 * torch.ones_like(self.w.value(state))))
        
        context.step('theta_1', [ self.theta_1 ], proposal=NormalProposal(self.theta_1.value(state), 0.5))
        context.step('theta_2', [ self.theta_2 ], proposal = NormalProposal(self.theta_2.value(state), 0.5))

        context.step('z', [ self.z ], proposal=CategoricalProposal(self.w.value(state).expand(self.N + self.w.value(state).shape)))

        context.step('x_1', [ self.x_1 ], proposal=NormalProposal(self.x_1.value(state), 1.0))
        context.step('x_2', [ self.x_2 ], proposal=NormalProposal(self.x_2.value(state), 3.0))

    #region theta_1

    # def log_prob_theta_1(self, state):
        
    #     # p(theta_1 | x_1) = p(theta_1) * p(x_1 | theta_1)

    #     lp = self.theta_1.log_prob(state)
    #     lp_x_1 = torch.where(self.z.value(state) == 0, self.x_1.log_prob(state), -torch.inf)
    #     lp += torch.logsumexp(lp_x_1, dim=0)
    #     return lp

    #endregion
    #region theta_2

    # def log_prob_theta_2(self, state):
        
    #     # p(theta_2 | x_2) = p(theta_2) * p(x_2 | theta_2)

    #     lp = self.theta_2.log_prob(state)
    #     lp_x_2 = torch.where(self.z.value(state) == 1, self.x_2.log_prob(state), -torch.inf)
    #     lp += torch.logsumexp(lp_x_2, dim=0)
    #     return lp

    #endregion
    #region z

    # def log_prob_z(self, state):

    #     # p(z | w, obs) = p(z | w) * p(x | z, theta) * p(obs | x)

    #     z = self.z.value(state)

    #     lp = self.z.log_prob(state)
        
    #     lp_1 = self.x_1.log_prob(state)
    #     lp_2 = self.x_2.log_prob(state)
    #     lp += torch.select([lp_1, lp_2], z)

    #     lp += self.obs.log_prob(state)

    #     return lp

    #endregion
    #region x_1

    # def log_prob_x_1(self, state):
        
    #     # p(x_1 | theta_1, obs) = p(x_1 | theta_1) * p(obs | x_1)
        
    #     lp = self.x_1.log_prob(state)
    #     lp += self.obs.log_prob(state)
        
    #     return lp

    #endregion
    #region x_2

    # def log_prob_x_2(self, state):
        
    #     # p(x_2 | theta_2, obs) = p(x_2 | theta_2) * p(obs | x_2)
        
    #     lp = self.x_2.log_prob(state)
    #     lp += self.obs.log_prob(state)
        
    #     return lp

    #endregion