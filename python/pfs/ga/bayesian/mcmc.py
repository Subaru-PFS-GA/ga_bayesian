from .constants import Constants
from .defaults import Defaults

class MCMC():
    def __init__(
            self,
            kernel,
            num_warmup=Defaults.mcmc_num_warmup,
            num_samples=Defaults.mcmc_num_samples,
            num_chains=Defaults.mcmc_num_chains,
        ):

        self.__kernel = kernel