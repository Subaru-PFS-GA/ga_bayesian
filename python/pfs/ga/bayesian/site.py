import torch

class Site():
    def __init__(self, name, parents=None, children=None, plates=None, selectors=None):
        self.__name = name
        self.__parents = list(parents) if parents is not None else []
        self.__children = list(children) if children is not None else []
        self.__plates = list(plates) if plates is not None else []
        self.__selectors = list(selectors) if selectors is not None else []

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.__name}')"

    #region Properties

    def __get_name(self):
        return self.__name
    
    name = property(__get_name)

    def __get_parents(self):
        return self.__parents
    
    parents = property(__get_parents)

    def __get_children(self):
        return self.__children
    
    children = property(__get_children)

    def __get_plates(self):
        return self.__plates
    
    plates = property(__get_plates)

    def __get_selectors(self):
        return self.__selectors
    
    selectors = property(__get_selectors)

    #endregion