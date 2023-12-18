import torch.nn as nn


class GELUTanh(nn.GELU):

    def __init__(self) -> None:
        super().__init__(approximate="tanh")
