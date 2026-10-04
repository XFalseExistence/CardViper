import hashlib

import pytest

from cardviper_ml.yolo_annotations import parse_yolo_row


def test_detection_and_polygon_enclosing_boxes():
    box = parse_yolo_row("0 .5 .5 .4 .2", 52)
    assert box.annotation_type == "bbox"
    assert box.bbox == pytest.approx((.3, .4, .7, .6))
    assert box.point_count is None
    triangle = parse_yolo_row("51 .1 .2 .9 .2 .5 .8", 52)
    assert triangle.annotation_type == "polygon"
    assert triangle.point_count == 3
    assert triangle.bbox == (.1, .2, .9, .8)
    polygon = parse_yolo_row("1 .1 .2 .4 .1 .9 .7 .1 .2", 52)
    assert polygon.point_count == 4
    assert polygon.bbox == (.1, .1, .9, .7)
    assert polygon.source_sha256 == hashlib.sha256(b"1 .1 .2 .4 .1 .9 .7 .1 .2").hexdigest()


@pytest.mark.parametrize("row", [
    "0 .1 .2 .3 .4 .5 .6 .7",  # odd coordinate count
    "0 .1 .2 0 .4",  # zero-width bbox, not a polygon
    "0 .1 .2 .3 .4 .5",  # fewer than three polygon points
    "0 .1 .2 .3 .4 .5 .6 .7 .8 .9",  # odd coordinate count
    "0 .1 .2 .3 .4 nan .6", "0 .1 .2 .3 .4 inf .6",
    "0 .1 .2 .3 .4 1.1 .6", "0 .1 .2 .1 .5 .1 .8",
    "52 .1 .2 .3 .4 .5 .6", "-1 .1 .2 .3 .4 .5 .6",
])
def test_malformed_rows_fail_closed(row):
    with pytest.raises(ValueError):
        parse_yolo_row(row, 52)
