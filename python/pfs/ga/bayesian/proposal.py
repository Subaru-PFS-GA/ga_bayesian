from .constants import Constants
from .defaults import Defaults

__all__ = ['Proposal']

class Proposal():
    """
    A base class for adaptive proposal distributions.

    Variables:
    ----------
    model: Model
        The model for which the proposal distribution is used.
    gamma: float
        Controls the memory of the adaptation.
    dist: torch.distributions.Distribution
        The distribution object.
    """

    def __init__(
            self,
            /,
            gamma=Constants.MISSING
        ):
        
        self._gamma = gamma if gamma is not Constants.MISSING else Defaults.proposal_gamma
        
        self.__adaptive = True
        self.__dist = self.__dist = self._create_dist_impl()

    #region Properties


    def __get_adaptive(self):
        return self.__adaptive
    
    def __set_adaptive(self, adaptive):
        self.__adaptive = adaptive

    adaptive = property(__get_adaptive, __set_adaptive)

    def __get_gamma(self):
        return self._gamma
    
    gamma = property(__get_gamma)

    def __get_batch_shape(self):
        return self.__dist.batch_shape
    
    batch_shape = property(__get_batch_shape)

    def __get_event_shape(self):
        return self.__dist.event_shape
    
    event_shape = property(__get_event_shape)

    #endregion

    def _create_dist_impl(self):
        """
        When implemented, creates the distribution object.
        """

        raise NotImplementedError()

    def update(self, x):
        """
        Updates the proposal distribution.

        Parameters:
        -----------
        x: array_like
            The new sample.
        """

        if self.__adaptive:
            self._update_impl(x)
            self.__dist = self._create_dist_impl()

    def _update_impl(self, x):
        """
        Updates the proposal distribution.

        Parameters:
        -----------
        x: array_like
            The new sample.
        """

        raise NotImplementedError()
    
    def sample(self, shape=()):
        """
        Draws a sample from the proposal distribution to be used as the next
        step in the MCMC chain.

        Parameters:
        -----------
        shape: tuple
            The shape of the sample to draw.
        """

        return self.__dist.sample(shape)