import torch
import torch.nn as nn

from .constants import GRU_ALL_LETTERS, GRU_HIDDEN_SIZE, GRU_N_LETTERS

# For backward compatibility with existing code
n_hidden = GRU_HIDDEN_SIZE
all_letters = GRU_ALL_LETTERS
n_letters = GRU_N_LETTERS


def infer(net: torch.nn.Module, name: str) -> torch.Tensor:
    net.eval()
    name_ohe = name_rep(name)
    hidden = net.init_hidden()  # type: ignore[attr-defined]
    output = None
    for i in range(name_ohe.size()[0]):
        output, hidden = net(name_ohe[i], hidden)

    return output  # type: ignore[return-value]


def name_rep(name: str) -> torch.Tensor:
    rep = torch.zeros(len(name), 1, n_letters)
    for index, letter in enumerate(name):
        pos = all_letters.find(letter)
        rep[index][0][pos] = 1
    return rep


class GRU_net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRU(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(
        self, input_: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.gru_cell(input_.view(1, 1, -1), hidden)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output.view(1, -1), hidden

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size)
