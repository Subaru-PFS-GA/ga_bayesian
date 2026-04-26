import torch

from pfs.ga.bayesian import Model, Variable, Proposal
from pfs.ga.bayesian.distributions import Normal
from pfs.ga.bayesian.proposals import NormalProposal

class Simple(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.theta = self.variable('theta', Normal(0.0, 1.0))
        self.x = self.variable('x', lambda state: Normal(self.theta.value(state), 1.0))
        self.obs = self.observed('obs', lambda state: Normal(self.x.value(state), 0.5))

    def sample(self, state):
        theta = self.theta.sample(state)
        x = self.x.sample(state)
        obs = self.obs.sample(state)
    
    def propose_x(self, init_state, final_state):
        x = self.proposal('x').sample()
        self.x.set(final_state, x)

    def update_x(self, final_state):
        x = self.x.value(final_state)
        self.proposal('x').update(x)

    def log_prob_x_given_all(self, state):
        lp = self.x.log_prob(state)
        lp += self.obs.log_prob(state)
        return lp

    def build(self, init_state):
        self.proposal('x', NormalProposal(self.x.value(init_state), 1.0))
        self.step('x', self.propose_x, self.update_x, self.log_prob_x_given_all)