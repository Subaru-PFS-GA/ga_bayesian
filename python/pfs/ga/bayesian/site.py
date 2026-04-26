import torch

class Site():
    def __init__(self, name):
        self.__name = name

    #region Properties

    def __get_name(self):
        return self.__name
    
    name = property(__get_name)

    #endregion