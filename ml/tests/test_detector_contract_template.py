import json
from pathlib import Path


def test_detector_template_is_not_a_production_contract():
    path = Path(__file__).parents[1] / "contracts/detector.template.json"
    data = json.loads(path.read_text())
    assert data["status"] == "TEMPLATE_ONLY"
    assert data["semantic_classes"] == ["CARD"]
    assert data["artifact"] is None
    assert data["output_tensors"] is None
    assert data["box_coordinate_format"] is None
    assert data["nms_ownership"] is None
    assert data["training_selection_provenance"] is None
