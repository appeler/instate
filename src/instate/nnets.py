"""Character-level neural network architecture and input encoding."""

import torch
import torch.nn as nn

from .constants import CHAR_TO_IDX


def canonicalize_name(name: str) -> str:
    """Return the exact lowercase ASCII representation consumed by current models."""
    return "".join(character for character in name.lower() if character in CHAR_TO_IDX)


def encode_name(name: str) -> list[int]:
    """Canonicalize a name and map it to model character indices."""
    return [CHAR_TO_IDX[character] for character in canonicalize_name(name)]


def pad_encoded(encoded: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad encoded names and return the tensor with their original lengths."""
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    maxlen = int(lengths.max()) if len(encoded) else 0
    x = torch.zeros(len(encoded), maxlen, dtype=torch.long)
    for i, e in enumerate(encoded):
        x[i, : len(e)] = torch.tensor(e, dtype=torch.long)
    return x, lengths


class CharBiLSTM(nn.Module):
    """Character-level bidirectional LSTM classifier.

    The model embeds a name, runs a packed bidirectional LSTM, and returns raw
    logits for each class. State and language prediction share this architecture.
    """

    def __init__(
        self,
        num_chars: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the classifier layers."""
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
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return class logits for a padded batch of encoded names."""
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n last two rows are the final fwd + bwd hidden states -> concat
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)


# State and language models share the same architecture.
StateLSTM = CharBiLSTM
