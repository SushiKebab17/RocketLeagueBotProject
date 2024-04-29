import torch.nn as nn

# An unused class incorporating the Tanh approximation of GELU


class GELUTanh(nn.GELU):

    def __init__(self) -> None:
        super().__init__(approximate="tanh")
