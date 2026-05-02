class Factor:
    """
    Container for one factor in a factor graph.

    It stores:
    - name: factor name
    - site: stochastic site whose conditional induces this factor
    - scope: stochastic sites participating in the factor
    """

    def __init__(self, name, site, scope):
        self.__name = name
        self.__site = site
        self.__scope = list(scope)

    #region Properties

    def __get_name(self):
        return self.__name

    name = property(__get_name)

    def __get_site(self):
        return self.__site

    site = property(__get_site)

    def __get_scope(self):
        return list(self.__scope)

    scope = property(__get_scope)

    #endregion


class FactorGraph:
    """
    Container for a factor graph derived from the traced Bayesian network.

    It stores:
    - sites: stochastic variable sites in graph order
    - factors: factors in graph order
    """

    def __init__(self, sites, factors):
        self.__sites = list(sites)
        self.__factors = list(factors)

    #region Properties

    def __get_sites(self):
        return list(self.__sites)

    sites = property(__get_sites)

    def __get_factors(self):
        return list(self.__factors)

    factors = property(__get_factors)

    #endregion

    def __iter__(self):
        return iter(self.__factors)

    def __len__(self):
        return len(self.__factors)
