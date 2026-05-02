import torch
from torch.distributions.constraints import real, simplex
from torch.distributions.constraints import interval, integer_interval

class Step():
    def __init__(
        self,
        name,
        sites, /,
        proposal,
        propose_func,
        update_func,
        log_prob_func,
        factor_sites=None,
        plate_dims=None,
    ):
        
        self.__name = name
        self.__sites = sites
        self.__proposal = proposal
        self.__propose_func = propose_func
        self.__update_func = update_func
        self.__log_prob_func = log_prob_func
        self.__factor_sites = list(factor_sites) if factor_sites is not None else []
        self.__plate_dims = dict(plate_dims) if plate_dims is not None else {}

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

    def __get_factor_sites(self):
        return list(self.__factor_sites)

    factor_sites = property(__get_factor_sites)

    def __get_plate_dims(self):
        return dict(self.__plate_dims)

    plate_dims = property(__get_plate_dims)

    #endregion
    
    def propose(self, current_state):
        # Sample from the proposal distribution
        step_state = {}
        self.__propose_func(self, step_state)

        # Add the proposed values to the step state
        for site in self.__sites:
            v = site.value(step_state)

            # If the distribution of the site is bounded,
            # we need to reflect the proposed value if it goes out of bounds.
            if site.dist.support is real:
                # Any unbounded distribution
                pass
            elif site.dist.support is simplex:
                # Dirichlet distribution.
                # When using a Dirichlet proposal, it should be fine.
                pass
            elif isinstance(site.dist.support, interval):
                # Reflect the value if it goes out of bounds
                lo = site.dist.support.lower_bound
                hi = site.dist.support.upper_bound
                w = hi - lo
                y = (v -lo) % (2 * w)
                v = lo + torch.where(y <= w, y, 2 * w - y)
            elif isinstance(site.dist.support, integer_interval):
                # Categorial distribution.
                # When using a Categorical proposal, it should be fine.
                pass
            else:
                raise ValueError(f"Unsupported distribution support for site '{site.name}'.")
            
            site.set(step_state, v)

        return step_state

    def update(self, final_state):
        return self.__update_func(self, final_state)

    def log_prob(self, state):
        return self.__log_prob_func(self, state)