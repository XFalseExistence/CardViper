from cardviper_ml.eval_classifier import evaluate_scores
from cardviper_ml.labels import LABELS


def _scores(label, confidence=.9):
    result = [(1 - confidence) / 52] * 53
    result[LABELS.index(label)] = confidence
    return result


def test_exact_top1_back_and_red_black_twos():
    report = evaluate_scores(["2C", "2D", "BACK"],
                             [_scores("2C"), _scores("2H"), _scores("BACK")], split="test")
    assert report["top1_accuracy"] == 2 / 3
    assert report["back"]["recall"] == 1
    assert report["back"]["precision"] == 1
    assert report["twos"]["red_vs_black_confusions"] == 0
    assert report["twos"]["confusion_matrix"]["2D"]["2H"] == 1
    assert report["confusion_matrix"]["2D"]["2H"] == 1


def test_holdout_evaluation_is_blocked():
    import pytest
    with pytest.raises(ValueError, match="holdout"):
        evaluate_scores(["AC"], [_scores("AC")], split="holdout")
