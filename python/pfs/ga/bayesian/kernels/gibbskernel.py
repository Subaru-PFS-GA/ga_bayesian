import torch
from ..kernel import Kernel
from ..deterministic import Deterministic

class GibbsKernel(Kernel):
    def __init__(self, model):
        super().__init__(model)

    def step(self, init_state, accept_counts=None):
        final_state = {}
    
        for name, step in self.model.steps.items():
            # Make a step proposal and compute the log probabilities for acceptance
            step_state = step.propose(init_state)
            lp_init = step.log_prob(init_state)
            lp_final = step.log_prob({ **init_state, **step_state })

            # Accept or reject the proposal, this will update step_state
            # to contain the accepted values
            mask = self.accept(init_state, step_state, lp_init, lp_final)

            # Track acceptance counts per instance
            if accept_counts is not None:
                if name in accept_counts:
                    accept_counts[name] = accept_counts[name] + mask.long()
                else:
                    accept_counts[name] = mask.long()

            # Update the proposal distributions
            step.update(step_state)

            final_state.update(step_state)

        # Combine the initial state with the final state to get the full state after
        # all Gibbs steps have been taken.
        combined_state = { **init_state, **final_state }

        # Recompute deterministic values after updates to keep derived state consistent.
        for site in self.model.sites.values():
            if isinstance(site, Deterministic):
                site.eval(combined_state)

        return combined_state

    def accept(self, init_state, step_state, lp_init, lp_final):
        # Assume final_state contains only the variables that are being updated
        lp_accept = lp_final - lp_init
        mask = torch.log(torch.rand(size=lp_accept.shape)) < lp_accept

        for key in step_state:
            value = step_state[key]
            extra_dims = value.dim() - mask.dim()
            expanded_mask = mask.reshape(mask.shape + (1,) * extra_dims).expand_as(value)
            step_state[key] = torch.where(expanded_mask, value, init_state[key])

        return mask
        