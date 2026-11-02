"""The Session A trainer must remain visibly separate from production evidence."""

import pytest

from cardviper_ml.labels import LABELS


def test_demo_training_metadata_is_never_releasable(tmp_path):
    """Removing a non-promotable marker or adding BACK must fail this test."""
    from cardviper_ml.train_session_a_demo_classifier import build_demo_metadata

    metadata = build_demo_metadata(
        data_root=tmp_path,
        seed=7,
        epochs=3,
        batch_size=4,
        train_sample_count=12,
        smoke_sample_count=2,
    )

    assert metadata["usage"] == "SESSION_A_DEMO_ONLY"
    assert metadata["independence"] == "NOT_INDEPENDENT"
    assert metadata["independent_evaluation"] is False
    assert metadata["supported_labels"] == list(LABELS[:-1])
    assert metadata["output_width"] == 52
    assert metadata["back_supported"] is False
    assert "BACK" not in metadata["supported_labels"]


def test_face_only_label_validation_rejects_back_and_unknown_labels():
    """A regression that trains a BACK/unknown class must fail this test."""
    from cardviper_ml.train_session_a_demo_classifier import validate_face_only_labels

    assert validate_face_only_labels(["AC", "10H", "KS"]) == ("AC", "10H", "KS")
    with pytest.raises(ValueError, match="face-up"):
        validate_face_only_labels(["AC", "BACK"])
    with pytest.raises(ValueError, match="canonical"):
        validate_face_only_labels(["AC", "JOKER"])
