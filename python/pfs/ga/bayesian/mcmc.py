from tqdm import tqdm

from .constants import Constants
from .defaults import Defaults
from .io import MemoryTrace

class MCMC():
    def __init__(
            self,
            kernel,
            num_chains = Defaults.mcmc_num_chains,
            num_warmup = Defaults.mcmc_num_warmup,
            num_samples = Defaults.mcmc_num_samples,
            thinning = Defaults.mcmc_thinning,
            progress = Defaults.mcmc_progress,
            trace = Constants.MISSING,
        ):

        self.__kernel = kernel
        self.__num_chains = num_chains
        self.__num_warmup = num_warmup
        self.__num_samples = num_samples
        self.__thinning = thinning
        self.__progress = progress
        self.__trace = trace if trace is not Constants.MISSING else MemoryTrace()

    #region Properties

    def __get_trace(self):
        return self.__trace
    
    trace = property(__get_trace)

    #endregion

    def __generate_init_state(self):
        # Generate the initial state for each chain by sampling from the prior
        state = self.__kernel.model.sample()
        return state
    
    def __set_observed(self, state, observed, batch_shape=()):
        # Set the observed variables in the model
        for key, value in observed.items():
            self.__kernel.model.sites[key].set(
                state,
                value.expand(batch_shape + value.shape).shape
            )

    def __wrap_in_progress_bar(self, iterable, label=None):
        if self.__progress:
            return tqdm(iterable, desc=label)
        else:
            return iterable

    def run(self, init_state=Constants.MISSING, observed=Constants.MISSING):

        # Rebuild the model with the specified number of chains
        self.__kernel.model.reset()
        self.__kernel.model.build(batch_shape=(self.__num_chains,))

        if init_state is Constants.MISSING:
            init_state = self.__generate_init_state()

        if observed is not Constants.MISSING:
            # TODO: figure out how to expand the observed variables to match
            #       the number of chains
            self.__set_observed(init_state, observed, batch_shape=(self.__num_chains,))

        for i in self.__wrap_in_progress_bar(range(self.__num_warmup), label="Warmup"):
            final_state = self.__kernel.step(init_state)
            init_state = final_state

        for i in self.__wrap_in_progress_bar(range(self.__num_samples), label="Sampling"):
            # Save the current state to the trace
            if i % self.__thinning == 0:
                self.__trace.append(init_state)

            final_state = self.__kernel.step(init_state)
            init_state = final_state
            
