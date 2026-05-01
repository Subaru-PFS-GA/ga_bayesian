import torch

from ..defaults import Defaults
from ..distributions import Normal
from ..proposal import Proposal


class NormalProposal(Proposal):
    """
    Adaptive univariate normal proposal distribution.
    """

    def __init__(
            self,
            loc,
            scale, /,
            eps=Defaults.proposal_eps,
            gamma=Defaults.proposal_gamma):

        self.__loc = loc
        self.__scale = scale
        self.__var = scale * scale

        self.__mcmc_scale = 2.38

        # Distribution origin is always zero
        self.__loc_zero = torch.zeros_like(self.__loc)

        self.__eps = eps

        super().__init__(gamma=gamma)

    #region Properties

    def __get_loc(self):
        return self.__loc

    loc = property(__get_loc)

    def __get_scale(self):
        return self.__scale

    scale = property(__get_scale)

    def __get_eps(self):
        return self.__eps

    eps = property(__get_eps)

    #endregion

    def _update_impl(self, x):
        # If a sample dimension is provided, collapse it before updating moments.
        while x.ndim > self.__loc.ndim:
            x = x.mean(0)

        delta = x - self.__loc
        self.__loc = self._gamma * self.__loc + (1 - self._gamma) * x
        self.__var = self._gamma * self.__var + (1 - self._gamma) * delta * (x - self.__loc)

        if self.__eps is not None:
            min_var = self.__eps * self.__eps
            self.__var = torch.clamp(self.__var, min=min_var)

        self.__scale = torch.sqrt(self.__var)

    def _create_dist_impl(self):
        return Normal(self.__loc_zero, self.__mcmc_scale * self.__scale)
