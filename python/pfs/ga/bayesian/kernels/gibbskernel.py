import torch
from ..kernel import Kernel
from ..deterministic import Deterministic

class GibbsKernel(Kernel):
    def __init__(self, model):
        super().__init__(model)

    def step(self, init_state):
        final_state = {}
    
        for name, step in self.model.steps.items():
            step_state = {}
            step.propose(step_state)
            lp_init = step.log_prob(init_state)
            lp_final = step.log_prob({ **init_state, **step_state })
            self.accept(init_state, step_state, lp_init, lp_final)
            step.update(step_state)
            final_state.update(step_state)

        # Recompute deterministic values after updates to keep derived state consistent.
        combined_state = { **init_state, **final_state }
        for site in self.model.sites.values():
            if isinstance(site, Deterministic):
                site.eval(combined_state)

        final_state.update({
            site.name: combined_state[site.name]
            for site in self.model.sites.values()
            if isinstance(site, Deterministic)
        })

        # TODO: record trace

        return { **init_state, **final_state }

    def accept(self, init_state, final_state, lp_init, lp_final):
        # Assume final_state contains only the variables that are being updated
        lp_accept = lp_final - lp_init
        mask = torch.log(torch.rand(size=lp_accept.shape)) < lp_accept
        for key in final_state:
            final_state[key] = torch.where(mask, final_state[key], init_state[key])
        