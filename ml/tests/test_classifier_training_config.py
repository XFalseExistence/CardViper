import pytest

from cardviper_ml.labels import LABELS


def paths(tmp_path):
    from cardviper_ml.classifier_config import ClassifierDataPaths
    return ClassifierDataPaths(
        train=tmp_path / "train.jsonl",
        validation=tmp_path / "val.jsonl",
        test=tmp_path / "test.jsonl",
        holdout=tmp_path / "holdout.jsonl",
    )


def test_classifier_config_locks_width_order_back_index_and_seed(tmp_path):
    from cardviper_ml.classifier_config import ClassifierTrainingConfig

    config = ClassifierTrainingConfig(
        labels=LABELS,
        declared_output_width=53,
        seed=9162026,
        data=paths(tmp_path),
    )

    assert config.output_width == 53
    assert config.labels == LABELS
    assert config.output_index_to_label == tuple(enumerate(LABELS))
    assert config.labels[52] == "BACK"
    assert config.seed == 9162026
    assert config.training_inputs == (config.data.train, config.data.validation)
    assert config.data.test not in config.training_inputs
    assert config.data.holdout not in config.training_inputs


@pytest.mark.parametrize("labels", [LABELS[:-1], LABELS + ("JOKER",), tuple(reversed(LABELS))])
def test_classifier_config_rejects_missing_extra_or_reordered_classes(tmp_path, labels):
    from cardviper_ml.classifier_config import ClassifierTrainingConfig
    with pytest.raises(ValueError, match="canonical"):
        ClassifierTrainingConfig(labels=labels, declared_output_width=len(labels), seed=1, data=paths(tmp_path))


@pytest.mark.parametrize("width", [0, 52, 54, None, 53.0])
def test_classifier_config_rejects_silent_output_width_drift(tmp_path, width):
    from cardviper_ml.classifier_config import ClassifierTrainingConfig
    with pytest.raises(ValueError, match="53"):
        ClassifierTrainingConfig(labels=LABELS, declared_output_width=width, seed=1, data=paths(tmp_path))


@pytest.mark.parametrize("seed", [None, True, 1.5, "42"])
def test_deterministic_seed_must_be_explicit_integer(tmp_path, seed):
    from cardviper_ml.classifier_config import ClassifierTrainingConfig
    with pytest.raises(ValueError, match="seed"):
        ClassifierTrainingConfig(labels=LABELS, declared_output_width=53, seed=seed, data=paths(tmp_path))


@pytest.mark.parametrize("overlap", ["validation", "test", "holdout"])
def test_train_path_cannot_overlap_other_splits(tmp_path, overlap):
    from cardviper_ml.classifier_config import ClassifierDataPaths
    values = {
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "val.jsonl",
        "test": tmp_path / "test.jsonl",
        "holdout": tmp_path / "holdout.jsonl",
    }
    values[overlap] = values["train"]
    with pytest.raises(ValueError, match="distinct"):
        ClassifierDataPaths(**values)


def test_split_directory_factory_uses_four_fixed_separate_manifests(tmp_path):
    from cardviper_ml.classifier_config import ClassifierDataPaths
    value = ClassifierDataPaths.from_split_directory(tmp_path)
    assert value == ClassifierDataPaths(
        train=tmp_path / "train.jsonl",
        validation=tmp_path / "val.jsonl",
        test=tmp_path / "test.jsonl",
        holdout=tmp_path / "holdout.jsonl",
    )
