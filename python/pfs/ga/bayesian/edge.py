class Edge:
    """
    Directed stochastic edge induced by Markov blanket traversal.

    role values are relative to the query site(s):
    - parent: source -> query
    - child: query -> target
    - coparent: source -> target_child
    """

    def __init__(self, source, target, role, selections=None):
        self.__source = source
        self.__target = target
        self.__role = role
        self.__selections = list(selections) if selections is not None else []

    #region Properties

    def __get_source(self):
        return self.__source

    source = property(__get_source)

    def __get_target(self):
        return self.__target

    target = property(__get_target)

    def __get_role(self):
        return self.__role

    role = property(__get_role)

    def __get_selections(self):
        return list(self.__selections)

    selections = property(__get_selections)

    def __get_has_selector(self):
        return len(self.__selections) > 0

    has_selector = property(__get_has_selector)

    #endregion