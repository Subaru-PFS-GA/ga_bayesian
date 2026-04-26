class Step():
    def __init__(
        self,
        name,
        sites, /,
        proposal,
        propose_func,
        update_func,
        log_prob_func,
    ):
        
        self.__name = name
        self.__sites = sites
        self.__proposal = proposal
        self.__propose_func = propose_func
        self.__update_func = update_func
        self.__log_prob_func = log_prob_func

    #region Properties

    def __get_name(self):
        return self.__name
    
    name = property(__get_name)

    def __get_sites(self):
        return self.__sites
    
    sites = property(__get_sites)

    def __get_proposal(self):
        return self.__proposal
    
    proposal = property(__get_proposal)

    #endregion
    
    def propose(self, init_state, final_state):
        return self.__propose_func(init_state, final_state)

    def update(self, final_state):
        return self.__update_func(final_state)

    def log_prob(self, state):
        return self.__log_prob_func(state)