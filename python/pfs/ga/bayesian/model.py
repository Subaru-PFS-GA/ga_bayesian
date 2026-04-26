from collections import OrderedDict

from .constants import Constants
from .defaults import Defaults
from .variable import Variable
from .observed import Observed
from .deterministic import Deterministic
from .proposal import Proposal
from .step import Step

class Model():

    def __init__(self, dtype=Defaults.dtype):
        self.__dtype = dtype

        self.__sites = OrderedDict()
        self.__proposals = OrderedDict()
        self.__steps = OrderedDict()

    #region Properties

    def __get_dtype(self):
        return self.__dtype
    
    dtype = property(__get_dtype)

    def __get_sites(self):
        return self.__sites
    
    sites = property(__get_sites)

    def __get_proposals(self):
        return self.__proposals
    
    proposals = property(__get_proposals)

    def __get_steps(self):
        return self.__steps
    
    steps = property(__get_steps)

    #endregion
        
    def variable(self, name, dist=Constants.MISSING):
        if name in self.__sites and dist is Constants.MISSING:
            return self.__sites[name]
        else:
            if dist is Constants.MISSING:
                raise ValueError(f"Variable '{name}' not found in the model.")
            
            self.__sites[name] = Variable(name, dist)
            return self.__sites[name]
    
    def observed(self, name, dist=Constants.MISSING):
        if name in self.__sites and dist is Constants.MISSING:
            return self.__sites[name]
        else:
            if dist is Constants.MISSING:
                raise ValueError(f"Observed variable '{name}' not found in the model.")
            
            self.__sites[name] = Observed(name, dist)
            return self.__sites[name]
    
    def deterministic(self, name, func=Constants.MISSING):
        if name in self.__sites and func is Constants.MISSING:
            return self.__sites[name]
        else:
            if func is Constants.MISSING:
                raise ValueError(f"Deterministic variable '{name}' not found in the model.")
            
            self.__sites[name] = Deterministic(name, func)
            return self.__sites[name]
    
    def proposal(self, name, /, proposal=Constants.MISSING):
        if name in self.__proposals and proposal is Constants.MISSING:
            return self.__proposals[name]
        else:
            if proposal is Constants.MISSING:
                raise ValueError(f"Proposal '{name}' not found in the model.")
            
            self.__proposals[name] = proposal
            return self.__proposals[name]

    def step(self, name, propose_func=Constants.MISSING, update_func=Constants.MISSING, log_prob_func=Constants.MISSING):
        if name in self.__steps and propose_func is Constants.MISSING and update_func is Constants.MISSING and log_prob_func is Constants.MISSING:
            return self.__steps[name]
        else:
            if propose_func is Constants.MISSING or update_func is Constants.MISSING or log_prob_func is Constants.MISSING:
                raise ValueError(f"Step '{name}' not found in the model.")
            
            self.__steps[name] = Step(name, propose_func, update_func, log_prob_func)
            return self.__steps[name]

    def build(self, init_state):
        raise NotImplementedError("The 'build' method must be implemented by the subclass.")