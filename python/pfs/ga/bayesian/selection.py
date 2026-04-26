from .deterministic import Deterministic


class Selection(Deterministic):
    """
    Deterministic site representing a selection operation.

    This site is created by context.select(...) so selection nodes can be
    identified explicitly in the graph while preserving deterministic behavior.
    """

    def __init__(self, name, eval_func, parents=None, children=None, plates=None, input_values=None, selector=None):
        super().__init__(name, eval_func, parents=parents, children=children, plates=plates)

        self.__input_values = input_values
        self.__selector = selector

    #region Properties

    def __get_input_values(self):
        return self.__input_values

    input_values = property(__get_input_values)

    def __get_selector(self):
        return self.__selector

    selector = property(__get_selector)

    #endregion
