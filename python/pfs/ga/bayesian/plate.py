class Plate():
    def __init__(self, name, size, stack=None):
        self.__name = name
        self.__size = size if isinstance(size, tuple) else (size,)
        self.__stack = stack

    #region Properties

    def __get_name(self):
        return self.__name

    name = property(__get_name)

    def __get_size(self):
        return self.__size

    size = property(__get_size)

    #endregion

    def set_stack(self, stack):
        self.__stack = stack

    def __enter__(self):
        if self.__stack is not None:
            self.__stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__stack is not None:
            if not self.__stack or self.__stack[-1] is not self:
                raise RuntimeError(f"Plate '{self.__name}' is not the innermost active plate.")
            self.__stack.pop()
        return False