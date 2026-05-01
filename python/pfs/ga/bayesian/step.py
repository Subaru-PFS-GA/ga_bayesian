import torch
from torch.distributions.constraints import real, interval

class Step():
    def __init__(
        self,
        name,
        sites, /,
        proposal,
        propose_func,
        update_func,
        log_prob_func,
    ):
        
        self.__name = name
        self.__sites = sites
        self.__proposal = proposal
        self.__propose_func = propose_func
        self.__update_func = update_func
        self.__log_prob_func = log_prob_func

    #region Properties

    def __get_name(self):
        return self.__name
    
    name = property(__get_name)

    def __get_sites(self):
        return self.__sites
    
    sites = property(__get_sites)

    def __get_proposal(self):
        return self.__proposal
    
    proposal = property(__get_proposal)

    #endregion
    
    def propose(self, current_state):
        # Sample from the proposal distribution
        step_state = {}
        self.__propose_func(self, step_state)

        # Add the proposed values to the step state
        for site in self.__sites:
            value = site.value(current_state)
            delta = step_state[site.name]

            v = value + delta

            # If the distribution of the site is bounded,
            # we need to reflect the proposed value if it goes out of bounds.
            if site.dist.support is real:
                pass
            elif isinstance(site.dist.support, interval):
                # Reflect the value if it goes out of bounds
                lo = site.dist.support.lower_bound
                hi = site.dist.support.upper_bound
                w = hi - lo
                y = (v -lo) % (2 * w)
                v = lo + torch.where(y <= w, y, 2 * w - y)
            else:
                raise ValueError(f"Unsupported distribution support for site '{site.name}'.")
            
            site.set(step_state, v)

        return step_state

    def update(self, final_state):
        return self.__update_func(self, final_state)

    def log_prob(self, state):
        return self.__log_prob_func(self, state)