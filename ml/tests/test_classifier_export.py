from cardviper_ml.export_classifier import write_labels
from cardviper_ml.labels import LABELS


def test_canonical_labels_file_is_exact(tmp_path):
    output = tmp_path / "labels.txt"
    write_labels(output)
    assert output.read_text().splitlines() == list(LABELS)
    assert output.read_text().splitlines()[52] == "BACK"
