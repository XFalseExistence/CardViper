import pytest

from cardviper_ml.inspect_litert import validate_tensor_report


def test_rejects_non_53_output_width():
    with pytest.raises(ValueError, match="53"):
        validate_tensor_report({"input_tensors": [{"shape": [1, 224, 224, 3]}],
                                "output_tensors": [{"shape": [1, 52]}]})
