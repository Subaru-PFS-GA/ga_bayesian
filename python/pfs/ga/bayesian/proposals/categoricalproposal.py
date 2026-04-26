import torch

from ..constants import Constants
from ..defaults import Defaults
from ..distributions import Categorical
from ..proposal import Proposal

class CategoricalProposal(Proposal):
    """
    Proposal distribution to generate steps for the Categorial distribution.
    """

    def __init__(
            self,
            w, /,
            gamma=Defaults.proposal_gamma):
        
        """
        Initialize the proposal distribution.
        
        Parameters:
        -----------
        w: array_like
            The initial weights.
        gamma: float
            Controls the memory of the adaptation.
        """

        self.__w = w / w.sum(-1)[..., None]

        super().__init__(gamma=gamma)

    #region Properties

    def __get_w(self):
        return self.__w
    
    def __set_w(self, w):
        self.__w = w / w.sum(-1)[..., None]
    
    w = property(__get_w, __set_w)

    #endregion

    def _update_impl(self, z):

        # Input is the categorical variable
        # Count items for all categories

        nw = torch.zeros_like(self.__w)
        for k in range(self.__w.shape[-1]):
            nw[..., k] = (z == k).sum(0) / z.shape[0]

        # Update the weight with finite memory
        self.__w = self._gamma * self.__w + (1 - self._gamma) * nw

    def _create_dist_impl(self):
        return Categorical(self.__w)
