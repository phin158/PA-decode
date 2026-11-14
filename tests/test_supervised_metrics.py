import _path  # noqa: F401
from pa_decode.supervised.metrics import self_check


def test_metrics_match_mstcn_eval():
    self_check(n=200)
