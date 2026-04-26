import torch

from pfs.ga.bayesian import Constants
from pfs.ga.bayesian import Model
from pfs.ga.bayesian import torch_extensions
from pfs.ga.bayesian.distributions import Uniform, Dirichlet, Categorical, Normal
from pfs.ga.bayesian.proposals import CategoricalProposal, NormalProposal

class Mixture(Model):

    def __init__(self, N=Constants.MISSING, C=Constants.MISSING, *args, **kwargs):
        super().__init__(*args, **kwargs)

        N = N if N is not Constants.MISSING else (100,)
        C = C if C is not Constants.MISSING else ()

        self.K = 2
        self.N = N
        self.C = C

        # p(w)
        self.w = self.variable('w', Dirichlet(5.0 * torch.ones(self.K)).expand(self.C))

        # p(theta_k) for each population
        self.theta_1 = self.variable('theta_1', lambda state: Uniform(-1.0, 1.0, validate_args=False).expand(self.C))
        self.theta_2 = self.variable('theta_2', lambda state: Uniform(2.0, 3.0, validate_args=False).expand(self.C))

        # p(z_i | w) for each data point
        self.z = self.variable('z', lambda state: Categorical(self.w.value(state)).expand(self.N + self.C))

        # p(x_ik | theta_k, z_i) for each data point and population
        self.x_1 = self.variable('x_1', lambda state: Normal(self.theta_1.value(state), 1.0).expand(self.N + self.C))
        self.x_2 = self.variable('x_2', lambda state: Normal(self.theta_2.value(state), 3.0).expand(self.N + self.C))

        # x_i = (x_i1, x_i2)[z_i] for each data point
        self.x = self.deterministic('x', self.eval_x)

        # p(obs_i | x_i) for each data point
        self.obs = self.observed('obs', lambda state: Normal(self.x.value(state), 0.25))
        
    def eval_x(self, state):
        z = self.z.value(state)
        x_1 = self.x_1.value(state)
        x_2 = self.x_2.value(state)
        x = torch.select([x_1, x_2], z)
        return x

    def sample(self, state):
        # Sample the parameters in topological order
        w = self.w.sample(state)
        
        z = self.z.sample(state)

        theta_1 = self.theta_1.sample(state)
        x_1 = self.x_1.sample(state)

        theta_2 = self.theta_2.sample(state)
        x_2 = self.x_2.sample(state)

        x = self.x.eval(state)

        obs = self.obs.sample(state)

    def build(self, init_state):
        # Create the proposals and steps for each group of sampled variables

        self.proposal_theta_1 = self.proposal('theta_1', NormalProposal(self.theta_1.value(init_state), 0.5))
        self.step('theta_1', self.propose_theta_1, self.update_theta_1, self.log_prob_theta_1)

        self.proposal_theta_2 = self.proposal('theta_2', NormalProposal(self.theta_2.value(init_state), 0.5))
        self.step('theta_2', self.propose_theta_2, self.update_theta_2, self.log_prob_theta_2)

        # Initialize the proposal for z to the initial weights
        # The weights vector need to be repeated for each data point
        z = self.z.value(init_state)
        w = self.w.value(init_state)
        w = w.expand(z.shape[:1] + w.shape)
        self.proposal_z = self.proposal('z', CategoricalProposal(w))
        self.step_z = self.step('z', self.propose_z, self.update_z, self.log_prob_z)

        self.proposal_x_1 = self.proposal('x_1', NormalProposal(self.x_1.value(init_state), 1.0))
        self.step_x_1 = self.step('x_1', self.propose_x_1, self.update_x_1, self.log_prob_x_1)

        self.proposal_x_2 = self.proposal('x_2', NormalProposal(self.x_2.value(init_state), 3.0))
        self.step_x_2 = self.step('x_2', self.propose_x_2, self.update_x_2, self.log_prob_x_2)
    
    #region theta_1

    def log_prob_theta_1(self, state):
        
        # p(theta_1 | x_1) = p(theta_1) * p(x_1 | theta_1)

        lp = self.theta_1.log_prob(state)
        lp_x_1 = torch.where(self.z.value(state) == 0, self.x_1.log_prob(state), -torch.inf)
        lp += torch.logsumexp(lp_x_1, dim=0)
        return lp
    
    def propose_theta_1(self, init_state, final_state):
        theta_1 = self.proposal_theta_1.sample()
        self.theta_1.set(final_state, theta_1)

    def update_theta_1(self, final_state):
        theta_1 = self.theta_1.value(final_state)
        self.proposal_theta_1.update(theta_1)

    #endregion
    #region theta_2

    def log_prob_theta_2(self, state):
        
        # p(theta_2 | x_2) = p(theta_2) * p(x_2 | theta_2)

        lp = self.theta_2.log_prob(state)
        lp_x_2 = torch.where(self.z.value(state) == 1, self.x_2.log_prob(state), -torch.inf)
        lp += torch.logsumexp(lp_x_2, dim=0)
        return lp

    def propose_theta_2(self, init_state, final_state):
        theta_2 = self.proposal_theta_2.sample()
        self.theta_2.set(final_state, theta_2)

    def update_theta_2(self, final_state):
        theta_2 = self.theta_2.value(final_state)
        self.proposal_theta_2.update(theta_2)

    #endregion
    #region z

    def log_prob_z(self, state):

        # p(z | w, obs) = p(z | w) * p(x | z, theta) * p(obs | x)

        z = self.z.value(state)

        lp = self.z.log_prob(state)
        
        lp_1 = self.x_1.log_prob(state)
        lp_2 = self.x_2.log_prob(state)
        lp += torch.select([lp_1, lp_2], z)

        lp += self.obs.log_prob(state)

        return lp

    def propose_z(self, init_state, final_state):
        z = self.proposal_z.sample()
        self.z.set(final_state, z)

    def update_z(self, final_state):
        z = self.z.value(final_state)
        self.proposal_z.update(z)

    #endregion
    #region x_1

    def log_prob_x_1(self, state):
        
        # p(x_1 | theta_1, obs) = p(x_1 | theta_1) * p(obs | x_1)
        
        lp = self.x_1.log_prob(state)
        lp += self.obs.log_prob(state)
        
        return lp

    def propose_x_1(self, init_state, final_state):
        x_1 = self.proposal_x_1.sample()
        self.x_1.set(final_state, x_1)

    def update_x_1(self, final_state):
        x_1 = self.x_1.value(final_state)
        self.proposal_x_1.update(x_1)

    #endregion
    #region x_2

    def log_prob_x_2(self, state):
        
        # p(x_2 | theta_2, obs) = p(x_2 | theta_2) * p(obs | x_2)
        
        lp = self.x_2.log_prob(state)
        lp += self.obs.log_prob(state)
        
        return lp

    def propose_x_2(self, init_state, final_state):
        x_2 = self.proposal_x_2.sample()
        self.x_2.set(final_state, x_2)

    def update_x_2(self, final_state):
        x_2 = self.x_2.value(final_state)
        self.proposal_x_2.update(x_2)

    #endregion