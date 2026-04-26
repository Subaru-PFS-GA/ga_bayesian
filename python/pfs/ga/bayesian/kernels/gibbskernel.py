import torch
from ..kernel import Kernel

class GibbsKernel(Kernel):
    def __init__(self, model):
        super().__init__(model)

    def step(self, state):
        init_state = state
        final_state = {}
    
        for name, step in self.model.steps.items():
            step_state = {}
            step.propose(init_state, step_state)
            lp_init = step.log_prob(init_state)
            lp_final = step.log_prob({ **init_state, **step_state })
            self.accept(init_state, step_state, lp_init, lp_final)
            step.update(step_state)
            final_state.update(step_state)

        # TODO: record trace

        return { **init_state, **final_state }

    def accept(self, init_state, final_state, lp_init, lp_final):
        # Assume final_state contains only the variables that are being updated
        lp_accept = lp_final - lp_init
        mask = torch.log(torch.rand(size=lp_accept.shape)) < lp_accept
        for key in final_state:
            final_state[key] = torch.where(mask, final_state[key], init_state[key])
        