import pytest

from cardviper_ml.detector_geometry import ResizeTransform


@pytest.mark.parametrize("width,height", [(640, 360), (360, 640), (320, 320),
                                           (2000, 180), (180, 2000)])
@pytest.mark.parametrize("policy", ["letterbox", "stretch"])
def test_round_trip_and_source_edges(width, height, policy):
    geometry = ResizeTransform(width, height, 320, 320, policy=policy)
    boxes = [(0, 0, width, height), (0, 0, 10, 10),
             (width - 10, height - 10, width, height)]
    for box in boxes:
        model = geometry.source_to_model(box)
        restored = geometry.model_to_source(model)
        assert restored == pytest.approx(box, abs=1e-6)
        assert 0 <= model[0] < model[2] <= 320
        assert 0 <= model[1] < model[3] <= 320


def test_letterbox_offsets_and_clamping():
    geometry = ResizeTransform(640, 320, 320, 320, policy="letterbox")
    assert geometry.pad_y == 80
    assert geometry.source_to_model((0, 0, 640, 320)) == pytest.approx((0, 80, 320, 240))
    assert geometry.model_to_source((-5, 70, 325, 250), clamp=True) == (0, 0, 640, 320)


def test_invalid_sizes_policy_and_box():
    with pytest.raises(ValueError):
        ResizeTransform(0, 10, 320, 320)
    with pytest.raises(ValueError):
        ResizeTransform(10, 10, 320, 320, policy="unknown")
    geometry = ResizeTransform(10, 10, 320, 320)
    with pytest.raises(ValueError):
        geometry.source_to_model((0, 0, 11, 5))
