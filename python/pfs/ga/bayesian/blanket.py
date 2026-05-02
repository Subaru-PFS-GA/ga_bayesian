class Blanket:
    """
    Container for a Markov blanket result.

    It stores:
    - sites: stochastic sites in the blanket
    - selections: Selection nodes traversed to recover the full blanket
    """

    def __init__(self, sites, selections=None):
        self.__sites = list(sites)
        self.__selections = list(selections) if selections is not None else []

    #region Properties

    def __get_sites(self):
        return list(self.__sites)

    sites = property(__get_sites)

    def __get_selections(self):
        return list(self.__selections)

    selections = property(__get_selections)

    def __get_has_selector(self):
        return len(self.__selections) > 0

    has_selector = property(__get_has_selector)

    #endregion

    def __iter__(self):
        return iter(self.__sites)

    def __len__(self):
        return len(self.__sites)

    def __contains__(self, item):
        return item in self.__sites
