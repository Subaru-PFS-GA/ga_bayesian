import torch
from torch.distributions import Distribution

from .site import Site

class Variable(Site):
    def __init__(self, name, dist, parents=None, children=None, plates=None):
        self.__dist = dist

        super().__init__(name, parents=parents, children=children, plates=plates)

    #region Properties

    def __get_dist(self):
        return self.__dist

    def __set_dist(self, dist):
        self.__dist = dist
    
    dist = property(__get_dist, __set_dist)

    #endregion

    def __resolve_dist(self, state):
        if isinstance(self.__dist, Distribution):
            return self.__dist
        elif callable(self.__dist):
            return self.__dist(state)
        else:
            raise ValueError(f"Invalid distribution for variable '{self.name}'.")

    def sample(self, state, *args, **kwargs):
        """
        Sample a value from the variable's distribution.

        Parameters:
        -----------
        args: tuple
            Positional arguments to be passed to the distribution's ``sample`` method.
        kwargs: dict
            Keyword arguments to be passed to the distribution's ``sample`` method.
        """

        dist = self.__resolve_dist(state)
        value = dist.sample(*args, **kwargs)
        state[self.name] = value
        return value
    
    def set(self, state, value):
        """
        Set the variable's value.

        Parameters:
        -----------
        value: any
            The new value for the variable.
        """

        state[self.name] = value
        return value

    def value(self, state):
        return state[self.name]
    
    def shape(self, state):
        return state[self.name].shape
    
    def log_prob(self, state):
        dist = self.__resolve_dist(state)
        return dist.log_prob(self.value(state))