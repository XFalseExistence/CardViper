from pathlib import Path
import re

import pytest


EXPECTED = tuple("AC AD AH AS 2C 2D 2H 2S 3C 3D 3H 3S 4C 4D 4H 4S "
                 "5C 5D 5H 5S 6C 6D 6H 6S 7C 7D 7H 7S 8C 8D 8H 8S "
                 "9C 9D 9H 9S 10C 10D 10H 10S JC JD JH JS QC QD QH QS "
                 "KC KD KH KS BACK".split())


def test_ordered_export_matches_android_vocabulary_and_output_indices(tmp_path):
    from cardviper_ml.labels import LABELS, write_labels, decode, encode
    assert LABELS == EXPECTED
    assert len(LABELS) == len(set(LABELS)) == 53
    assert LABELS.count("BACK") == 1
    root = Path(__file__).resolve().parents[2]
    codec = (root / "app/src/main/java/com/cardviper/app/vision/CardLabelCodec.kt").read_text()
    ranks = re.findall(r'CardRank\.\w+ to "([^"]+)"', codec)
    suits = re.findall(r'CardSuit\.\w+ to "([^"]+)"', codec)
    assert len(ranks) == 13 and len(suits) == 4
    assert set(LABELS) == {r + s for r in ranks for s in suits} | {"BACK"}
    output = tmp_path / "labels.txt"
    write_labels(output)
    assert output.read_text() == "\n".join(EXPECTED) + "\n"
    for index, label in enumerate(output.read_text().splitlines()):
        assert label == EXPECTED[index]
        assert encode(decode(label)) == label


def test_back_is_not_a_face():
    from cardviper_ml.labels import Back, Face, decode
    assert isinstance(decode("BACK"), Back)
    assert not isinstance(decode("BACK"), Face)
    assert decode("10H") == Face("10", "H")


@pytest.mark.parametrize("label", ["", "back", "ac", "Ac", "AC ", " AC", "1C", "TC", "11S", "A♠", "CARD", "BACKS", None])
def test_malformed_and_alias_labels_rejected(label):
    from cardviper_ml.labels import decode
    with pytest.raises(ValueError):
        decode(label)


def test_invalid_face_cannot_be_encoded():
    from cardviper_ml.labels import Face
    with pytest.raises(ValueError):
        Face("BACK", "S")
