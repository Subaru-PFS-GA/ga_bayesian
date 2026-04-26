from .site import Site


class Deterministic(Site):
    """
    Deterministic site in a Bayesian network.

    The site stores a set of input variables and a callable that combines
    their values into a derived deterministic value.
    """

    def __init__(self, name, eval_func):
        super().__init__(name)

        self.__eval_func = eval_func

    def eval(self, state):
        """
        Calculate the deterministic value based on the current state.

        Parameters:
        -----------
        state: dict
            The current state of the model, containing values for all variables.

        Returns:
        --------
        The calculated deterministic value.
        """

        value = self.__eval_func(state)
        state[self.name] = value
        return value

    def set(self, state):
        """
        Set the variable's value.

        Parameters:
        -----------
        value: any
            The new value for the variable.
        """

        value = self.__eval_func(state)
        state[self.name] = value
        return value

    def value(self, state):
        return state[self.name]