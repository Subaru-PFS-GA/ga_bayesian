import torch

from ..constants import Constants
from ..defaults import Defaults
from ..distributions import Dirichlet
from ..proposal import Proposal

class DirichletProposal(Proposal):
    """
    Proposal distribution to generate steps for the Dirichlet distribution.
    """

    def __init__(
            self,
            alpha, /,
            m = Constants.MISSING,
            gamma = Constants.MISSING
        ):

        """
        Initialize the proposal distribution.
        
        Parameters:
        -----------
        alpha: array_like
            The initial concentration parameter.
        m: float
            Maximum concentration.
        gamma: float
            Controls the memory of the adaptation.
        """

        # Concentration parameter
        self.__alpha = alpha                    
        
        # Maximum concentration
        self.__m = m if m is not Constants.MISSING else Defaults.proposal_dirichlet_max_concentration

        self.__reset_streaming_moments(alpha)

        super().__init__(gamma=gamma)

    def __reset_streaming_moments(self, alpha):
        # Initialize the streaming average and variance of the weights
        # to the expectation value and the variance of the Dirichlet
        # distribution with the initial alpha
        alpha_0 = alpha.sum(-1)[..., None]
        alpha_m = alpha / alpha_0
        self.__w_mean = alpha_m
        self.__w_var = alpha_m * (1 - alpha_m) / (alpha_0 + 1)

    #region Properties

    def __get_alpha(self):
        return self.__alpha
    
    def __set_alpha(self, alpha):
        self.__alpha = alpha
        self.__reset_streaming_moments(alpha)
    
    alpha = property(__get_alpha, __set_alpha)

    def __get_m(self):
        return self.__m
    
    def __set_m(self, m):
        self.__m = m
    
    m = property(__get_m, __set_m)

    #endregion

    def _update_impl(self, w):
        # Calculate the streaming average and variance of the weights with finite memory
        nw = w - self.__w_mean
        self.__w_mean = self._gamma * self.__w_mean + (1 - self._gamma) * w
        self.__w_var = self._gamma * self.__w_var + (1 - self._gamma) * nw * (w - self.__w_mean)

        # Solve for the new alpha
        alpha_0 = (self.__w_mean * (1 - self.__w_mean) / self.__w_var - 1).mean(-1)[..., None]
        na = self.__w_mean * alpha_0

        # Make sure the concentration is not too high
        if self.__m is not None:
            mx = na.max(-1).values[..., None]
            na = torch.where(mx > self.__m, na / mx * self.__m, na)
        
        self.__alpha = na
        
    def _create_dist_impl(self):
        return Dirichlet(self.__alpha)