class Kernel():
    def __init__(self, model):
        self.__model = model

    #region Properties

    def __get_model(self):
        return self.__model
    
    model = property(__get_model)

    #endregion