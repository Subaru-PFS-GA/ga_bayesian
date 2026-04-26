import torch
from torch.distributions import Distribution
from torch.distributions import constraints

__all__ = ["Delta"]

class Delta(Distribution):
    """
    Dirac delta distribution.
    """

    support = constraints
    has_rsample = True

    def __init__(self, loc, validate_args=None):

        self.__loc = loc
        batch_shape, event_shape = loc.shape[:-1], loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    #region Properties

    def __get_loc(self):
        return self.__loc
    
    loc = property(__get_loc)

    def __get_mean(self):
        return self.__loc

    mean = property(__get_mean)

    def __get_mode(self):
        return self.__loc

    mode = property(__get_mode)

    def __get_variance(self):
        return torch.zeros_like(self.__loc)

    variance = property(__get_variance)

    #endregion

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError()

    def rsample(self, sample_shape=()):
        shape = self._extended_shape(sample_shape)
        loc = self.__loc.expand(shape)
        return loc
    
    def log_prob(self, value):
        return torch.where((value == self.__loc).all(-1), 0.0, -torch.inf)
    
    def entropy(self):
        return torch.zeros_like(self.__loc[..., 0])
