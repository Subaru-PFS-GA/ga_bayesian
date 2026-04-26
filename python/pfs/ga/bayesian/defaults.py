import torch

class Defaults():
    dtype = torch.float32
    proposal_eps = 1e-4
    proposal_gamma = 0.99
    proposal_dirichlet_max_concentration = 5.0
    mcmc_num_warmup = 1000
    mcmc_num_samples = 1000
    mcmc_num_chains = 4
    mcmc_thinning = 1
    mcmc_progress = True
    trace_initial_size = 1000