import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import tsmom_backtest as ts  # noqa: E402


def _closes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = {s: 100 * np.cumprod(1 + rng.normal(d, 0.03, n))
            for s, d in [("BTC/USD", 0.002), ("ETH/USD", 0.001),
                         ("SOL/USD", 0.0015), ("LTC/USD", -0.002)]}
    return pd.DataFrame(cols, index=idx)


def test_weights_capped_and_no_negative_momentum():
    c = _closes()
    w = ts.target_weights(c, 300)
    assert (w <= ts.MAX_WEIGHT + 1e-12).all() and (w > 0).sum() <= ts.TOP_N
    assert (ts.target_weights(c, 10) == 0).all()


def test_no_lookahead():
    c = _closes()
    base = ts.run_backtest(c)["daily"]
    c2 = c.copy()
    c2.iloc[-1] *= 5
    alt = ts.run_backtest(c2)["daily"]
    assert np.allclose(base.iloc[:-1], alt.iloc[:-1])


def test_evaluate_criteria_keys():
    out = ts.evaluate_criteria(_closes(600))
    assert out["verdict"] in ("PASS", "REJECT")
    assert set(out["sensitivity_sharpe"]) == set(ts.SENSITIVITY_WEEKS)
