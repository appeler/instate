import torch
import torch.nn as nn

from .constants import (
    CHAR_TO_IDX,
    GRU_ALL_LETTERS,
    GRU_HIDDEN_SIZE,
    GRU_N_LETTERS,
)

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


def encode_name(name: str) -> list[int]:
    """Map a (cleaned, lowercase) name to ``CHAR_TO_IDX`` indices, dropping out-of-vocab chars."""
    return [CHAR_TO_IDX[c] for c in name if c in CHAR_TO_IDX]


def pad_encoded(encoded: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of index-lists into ``(LongTensor [B, T], lengths LongTensor [B])`` (PAD=0)."""
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    maxlen = int(lengths.max()) if len(encoded) else 0
    x = torch.zeros(len(encoded), maxlen, dtype=torch.long)
    for i, e in enumerate(encoded):
        x[i, : len(e)] = torch.tensor(e, dtype=torch.long)
    return x, lengths


class StateLSTM(nn.Module):
    """Char-level bidirectional LSTM for state prediction (v1.2.0, replaces ``GRU_net``).

    Embedding -> packed BiLSTM -> Linear over states. Mirrors the language ``LanguagePredictor``
    pattern. Trained/served with the 27-char ``CHAR_TO_IDX`` vocab (``<PAD>`` = 0). Outputs raw
    logits over ``GT_KEYS`` (use softmax/topk downstream).
    """

    def __init__(
        self,
        num_chars: int,
        num_states: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.embedding = nn.Embedding(num_chars, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(2 * hidden_dim, num_states)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (2, batch, hidden) for a 1-layer BiLSTM -> concat last fwd + bwd states
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)
