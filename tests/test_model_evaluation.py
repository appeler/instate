"""Tests for the metrics reported by the model-training programs."""

import pytest
import torch

from instate.constants import NUM_LANGUAGES
from model_training.train_lang_lstm import evaluate as evaluate_language
from model_training.train_state_lstm import evaluate as evaluate_state


class FixedRankModel(torch.nn.Module):
    """Return fixed rankings selected by the first encoded character."""

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Rank 0/1/2 for token 1 and 1/0/2 for token 2."""
        logits = torch.full((len(x), NUM_LANGUAGES), -100.0)
        for row, token in enumerate(x[:, 0].tolist()):
            order = (0, 1, 2) if token == 1 else (1, 0, 2)
            for score, label in enumerate(reversed(order), start=1):
                logits[row, label] = float(score)
        return logits


def test_state_evaluation_reports_modal_and_distribution_metrics() -> None:
    """State evaluation distinguishes modal hits from covered voter mass."""
    test = [([1], {0: 8, 3: 2}), ([2], {2: 6, 1: 4})]

    metrics = evaluate_state(FixedRankModel(), test, "cpu", batch_size=2)

    assert metrics == pytest.approx(
        {
            "modal_top1": 0.5,
            "modal_top3": 1.0,
            "mass_top1": 0.6,
            "mass_top3": 0.9,
        }
    )


def test_language_evaluation_reports_modal_and_distribution_metrics() -> None:
    """Language evaluation weights probability coverage by surname mass."""
    first = [0.0] * NUM_LANGUAGES
    first[0], first[3] = 0.8, 0.2
    second = [0.0] * NUM_LANGUAGES
    second[1], second[2] = 0.4, 0.6

    metrics = evaluate_language(
        FixedRankModel(), [[1], [2]], [first, second], [10.0, 10.0], "cpu"
    )

    assert metrics == pytest.approx(
        {
            "modal_top1": 0.5,
            "modal_top3": 1.0,
            "mass_top1": 0.6,
            "mass_top3": 0.9,
        }
    )
