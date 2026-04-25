import numpy as np
import torch

from ..distributions import MultivariateNormal
from .proposal import Proposal

class MultivariateNormalProposal(Proposal):
    """
    Proposal distribution to generate steps adaptively for any combination of variables.
    The posterior distribution is approximated by a multivariate normal.
    """

    def __init__(self, loc, cov, eps=1e-4, gamma=0.99):
        """
        Initialize the proposal distribution.

        Parameters:
        -----------
        gamma: float
            Controls the memory of the adaptation.
        loc: array_like
            The initial mean vector.
        cov: array_like
            The initial covariance matrix.
        eps: float
            Minimum step size, added to the diagonal of the covariance matrix.
        """

        self.__loc = loc                    # Current mean vector
        self.__cov = cov                    # Current covariance matrix

        self.__dim = self.__loc.shape[-1]

        # Scale
        self.__scale = 2.38 / np.sqrt(self.__dim)

        # Distribution origin is always zero
        self.__loc_zero = torch.zeros_like(self.__loc)

        # Cholesky decomposition of the covariance matrix
        self.__chol = torch.linalg.cholesky(cov)

        # Minimum step size, added to the diagonal of the covariance matrix
        if eps is not None:
            # Create a diagonal matrix with the minimum step size and
            # tile it to the shape of the covariance matrix
            self.__eps = torch.diag(torch.full((self.__dim,), eps, device=self.__loc.device, dtype=self.__loc.dtype))
            self.__eps = self.__eps.repeat(loc.shape[:-1] + (1, 1))

        super().__init__(gamma=gamma)

    #region Properties

    def __get__loc(self):
        return self.__loc
    
    loc = property(__get__loc)

    def __get_cov(self):
        return self.__cov
    
    cov = property(__get_cov)

    def __get_chol(self):
        return self.__chol
    
    chol = property(__get_chol)

    def __get_eps(self):
        return self.__eps
    
    eps = property(__get_eps)

    #endregion

    def __update_loc(self, loc, x, gamma):
        loc[()] = gamma * loc + (1 - gamma) * x

    def __update_cov(self, cov, loc, x, gamma):
        nn = x - loc
        # outer product of nn with itself
        nd = nn[..., None] * nn[..., None, :]     # Outer product
        cov[()] = gamma * cov + (1 - gamma) * nd

    def __update_chol(self, chol, loc, x, gamma):
        """
        Update the Cholesky decomposition of the covariance matrix using the rank-1 update algorithm.

        Parameters:
        -----------
        chol: array_like
            The current Cholesky decomposition of the covariance matrix.
        loc: array_like
            The current mean vector.
        x: array_like
            The current vector of the variables.

        Iterate over the dimensions of the covariance matrix and update the Cholesky decomposition
        using the rank-1 update algorithm. This is more efficient than calculating the Cholesky
        decomposition from scratch after each update. See:
        https://www.mathworks.com/matlabcentral/answers/498730-chol-update-algoryhtm-explanation
        """

        n = chol.shape[-1]
        x = (x - loc)[..., None]

        chol *= np.sqrt(gamma)
        x *= np.sqrt(1 - gamma)

        for k in range(n):
            r = torch.sqrt(chol[..., k, k]**2 + x[..., k, 0]**2)
            c = (r / chol[..., k, k])[..., None]
            s = (x[..., k, 0] / chol[..., k, k])[..., None]
            chol[..., k, k] = r
            chol[..., k + 1:, k] = (chol[..., k + 1:, k] + s * x[..., k + 1:, 0]) / c
            x[..., k + 1:, 0] = c * x[..., k + 1:, 0] - s * chol[..., k + 1:, k]

    def _update_impl(self, x):
        """
        Update step proposal covariance matrix and mean vector

        Parameters:
        -----------
        x: array_like
            The current vector of the variables.
        """

        if False:
            self.__update_loc(self.__loc, x, self._gamma)
            self.__update_cov(self.__cov, self.__loc, x, self._gamma)    
        else:
            self.__update_loc(self.__loc, x, self._gamma)
            self.__update_chol(self.__chol, self.__loc,  x, self._gamma)

    def _create_dist_impl(self):
        """
        Calculate the proposal distribution based on the current mean vector and
        covariance matrix and instantiate the distribution object.
        """

        if False:
            cov = self.__cov

            # Add minimum step size to the diagonal of the covariance matrix
            if self.__eps is not None:
                cov = cov + self.__eps**2

            return MultivariateNormal(self.__loc_zero, self.__scale * self.__scale * cov)
        else:
            chol = self.__chol

            # if self.__eps is not None:
            #     chol = self.__chol + self.__eps
            # else:
            #     chol = self.__chol

            # TODO: multiply by 2.38 * sqrt(dim) to get the right covariance matrix
            return MultivariateNormal(self.__loc_zero, scale_tril=(self.__scale * chol))