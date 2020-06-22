from .ratios import *
from .levels import extend_variable_pattern


def test_should_extend_empty_pattern():
    pattern = []

    patterns = extend_variable_pattern(pattern)
    patterns = list(patterns)

    assert patterns == [
        [RATIO_2_TO_1],
        [RATIO_3_TO_2],
        [RATIO_4_TO_3],
        [RATIO_5_TO_4],
        [RATIO_6_TO_5],
    ]


def test_should_extend_non_empty_pattern():
    pattern = [RATIO_3_TO_2]

    patterns = extend_variable_pattern(pattern)
    patterns = list(patterns)

    assert patterns == [
        [RATIO_3_TO_2, RATIO_2_TO_1],
        [RATIO_3_TO_2, RATIO_3_TO_2],
        [RATIO_3_TO_2, RATIO_4_TO_3],
        [RATIO_3_TO_2, RATIO_5_TO_4],
    ]
