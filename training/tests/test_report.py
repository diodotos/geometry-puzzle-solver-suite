from pathlib import Path

from gpss_train.report import main

SAMPLE = Path(__file__).with_name("eval_sample.json")


def test_tables_render():
    heuristic = main([SAMPLE], "heuristic").splitlines()
    assert heuristic[0].startswith("| true depth") and heuristic[-1].startswith("| all | 3,674,160")
    depth = main([SAMPLE], "depth").splitlines()
    assert len(depth) == 2 + 12 + 1
    assert "0.6 / 100" in main([SAMPLE, SAMPLE], "settings")


def test_summary():
    import json
    s = json.loads(main([SAMPLE], "summary"))
    assert s["lambda"] == 0.6 and s["solve_stratified"]["solved_rate"] == 1.0
