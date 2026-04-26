import torch

from ..defaults import Defaults
from ..constants import Constants
from .trace import Trace

class MemoryTrace(Trace):
    def __init__(self):
        super().__init__()

        self.__device = 'cpu'        
        self.__initial_size = Defaults.trace_initial_size
        self.__size = 0
        self.__items = None

    def __getitem__(self, key):
        return self.__items[key][:self.__size]

    def __len__(self):
        return self.__size

    def keys(self):
        return self.__items.keys()
    
    def __iter__(self):
        return iter(self.__items)
    
    def items(self):
        return self.__items.items()
    
    def values(self):
        return self.__items.values()

    def __init_trace(self, state):
        # Initialize the tensors to store the trace with the initial size
        # The trace is stored in the main memory so do not use torch but numpy
        self.__items = {}
        for key, value in state.items():
            value = value.cpu()
            self.__items[key] = torch.empty(
                (self.__initial_size,) + value.shape,
                dtype=value.dtype,
                device=self.__device)

    def append(self, state):
        if self.__items is None:
            self.__init_trace(state)

        for key, value in state.items():
            self.__items[key][self.__size] = value.cpu()

        self.__size += 1