import csv
import copy
import json
from types import SimpleNamespace

import pytest

from backtest import run_baseline as baseline


def write_prices(path, timestamps):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "close"])
        writer.writeheader()
        for timestamp in reversed(timestamps):
            writer.writerow({"timestamp": timestamp, "close": timestamp + 1})


def test_date_parsing_uses_utc():
    assert baseline.parse_date("1970-01-01") == 0
    assert baseline.parse_date("1970-01-01T01:00:00+01:00") == 0
    assert baseline.parse_date("1970-01-01T00:00:00Z") == 0


def test_filter_is_start_inclusive_end_exclusive(tmp_path):
    path = tmp_path / "BTC-USD_1min.csv"
    write_prices(path, [0, 60, 180, 240])
    assert baseline.load_csv_prices(str(path)) == [1, 61, 181, 241]
    coverage = {}
    assert baseline.load_csv_prices(str(path), 60, 240, coverage) == [61, 181]
    assert coverage["coverage_pct"] == pytest.approx(200 / 3)
    assert coverage["largest_gap_minutes"] == 2
    assert coverage["start_timestamp"] == 60
    assert coverage["end_timestamp"] == 180


def test_empty_window_and_duplicate_coverage(tmp_path):
    path = tmp_path / "BTC-USD_1min.csv"
    write_prices(path, [60, 60, 120])
    coverage = {}
    baseline.load_csv_prices(str(path), 0, 180, coverage)
    assert coverage["unique_minutes"] == 2
    assert coverage["coverage_pct"] == pytest.approx(200 / 3)
    assert baseline.load_csv_prices(str(path), 180, 240, coverage) == []
    assert coverage["coverage_pct"] == 0
    assert coverage["start_timestamp"] is None


def test_directory_filters_before_minimum_length_check(tmp_path):
    path = tmp_path / "BTC-USD_1min.csv"
    write_prices(path, range(0, 240 * 60, 60))
    coverage = {}
    assert baseline.load_price_dir(str(tmp_path), start_ts=60 * 100, coverage=coverage) == {}
    assert coverage["BTC-USD"]["rows"] == 140


@pytest.mark.parametrize(
    "meta,start,end,expected",
    [
        (None, 600, 1200, "unknown"),
        ({}, 600, 1200, "unknown"),
        ({"actual_data_end": "1970-01-01T00:10:00Z"}, 601, 1200, "out_of_sample"),
        ({"actual_data_end": "1970-01-01T00:10:00Z"}, 600, 1200, "unknown"),
        ({"actual_data_start": "1970-01-01", "actual_data_end": "1970-01-01T00:10:00Z"},
         600, 1200, "in_sample"),
        ({"actual_data_start": "1970-01-02", "actual_data_end": "1970-01-03"},
         0, 600, "unknown"),
    ],
)
def test_leakage_labels(meta, start, end, expected):
    coverage = {"spot": {"BTC": {"start_timestamp": start, "end_timestamp": end, "rows": 200}}}
    assert baseline.classify_window(meta, coverage) == expected


def test_overlapping_futures_prevents_out_of_sample_label():
    meta = {"actual_data_start": "1970-01-01", "actual_data_end": "1970-01-01T00:10:00Z"}
    coverage = {
        "spot": {"BTC": {"start_timestamp": 660, "end_timestamp": 1200, "rows": 200}},
        "futures": {"PI_XBTUSD": {"start_timestamp": 600, "end_timestamp": 1200, "rows": 200}},
    }
    assert baseline.classify_window(meta, coverage) == "in_sample"
    assert baseline.classify_window(meta, {"spot": {}}) == "unknown"


def test_aggregate_pools_trades_without_claiming_portfolio_equity():
    results = [
        SimpleNamespace(
            trades=[SimpleNamespace(pnl_usd=pnl, exit_reason=reason)],
            total_return_usd=pnl, max_drawdown_pct=dd, sharpe_ratio=sharpe,
        )
        for pnl, reason, dd, sharpe in [(10, "TAKE_PROFIT", 2, 1), (-5, "STOP_LOSS", 4, -1)]
    ]
    stats = baseline.aggregate_stats(results)
    assert stats["num_trades"] == 2
    assert stats["win_rate"] == 0.5
    assert stats["profit_factor"] == 2
    assert stats["total_return_usd"] == 5
    assert stats["worst_symbol_max_drawdown_pct"] == 4
    assert stats["mean_symbol_sharpe_ratio"] == 0
    assert stats["exit_reasons"] == {"TAKE_PROFIT": 1, "STOP_LOSS": 1}
    assert baseline.aggregate_stats([])["profit_factor"] == 0


@pytest.mark.parametrize("strict,training_end,should_run", [
    ("strict", "1970-01-01", True),
    ("strict", "1970-01-02", False),
    ("base", "1970-01-01", False),
])
def test_cli_out_of_sample_gate(tmp_path, monkeypatch, strict, training_end, should_run):
    spot = tmp_path / "spot"
    spot.mkdir()
    write_prices(spot / "BTC-USD_1min.csv", range(60, 241 * 60, 60))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"actual_data_start": "1970-01-01", "actual_data_end": training_end}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(baseline, "SPOT_DIR", str(spot))
    monkeypatch.setattr(baseline, "FUTURES_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(baseline.MarketPredictor, "load_model", lambda self: True)
    strict_config = copy.copy(baseline.live_config)
    strict_config.SIM_REALISM_PROFILE = "strict"
    strict_config._apply_realism_profile()
    for name, value in baseline.execution_settings(strict_config).items():
        monkeypatch.setattr(baseline.live_config, name, value)
    monkeypatch.setattr(baseline.live_config, "SIM_REALISM_PROFILE", strict)
    calls = []
    monkeypatch.setattr(baseline, "run_full_backtest", lambda *a, **kw: calls.append(kw) or {})
    monkeypatch.setattr(baseline, "print_aggregate_summary", lambda results: None)
    monkeypatch.setattr("sys.argv", ["run_baseline.py", "--model-meta", str(meta),
                                    "--output", str(output), "--require-out-of-sample"])
    if should_run:
        baseline.main()
        report = json.loads(output.read_text())
        assert report["evaluation_type"] == "out_of_sample"
        assert report["effective_strict_execution"]
        assert report["data_coverage"]["spot"]["BTC-USD"]["coverage_pct"] == 100
        assert calls[0]["price_data"]["BTC-USD"][0] == 61
    else:
        with pytest.raises(SystemExit) as error:
            baseline.main()
        assert error.value.code == 2
        assert calls == []
        assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("ENABLE_EXECUTION_COSTS", False),
    ("ENABLE_PARTIAL_FILLS", False),
    ("ENABLE_FUNDING_COSTS", False),
    ("SPOT_SLIPPAGE_BPS", 0),
    ("FUTURES_SLIPPAGE_BPS", 0),
    ("SPOT_FEE_RATE", 0),
    ("FUTURES_FEE_RATE", 0),
    ("PARTIAL_FILL_PROB", 0),
    ("PARTIAL_FILL_MIN", 1),
    ("PARTIAL_FILL_MAX", 1),
    ("FUTURES_FUNDING_RATE_PER_8H", 0),
    ("SPOT_SLIPPAGE_BPS", float("nan")),
    ("FUTURES_FEE_RATE", float("inf")),
])
def test_strict_gate_rejects_weakened_or_invalid_overrides(monkeypatch, field, value):
    strict_config = copy.copy(baseline.live_config)
    strict_config.SIM_REALISM_PROFILE = "strict"
    strict_config._apply_realism_profile()
    monkeypatch.setattr(baseline, "live_config", strict_config)
    assert baseline.strict_execution_enabled()
    monkeypatch.setattr(strict_config, field, value)
    assert not baseline.strict_execution_enabled()
    assert getattr(strict_config, field) is value


@pytest.mark.parametrize("args", [
    ["--start-date", "2026-10-03", "--end-date", "2026-10-02"],
    ["--candle-size", "0"],
    ["--start-date", "not-a-date"],
])
def test_cli_rejects_invalid_windows_before_loading_model(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["run_baseline.py", *args])
    with pytest.raises(SystemExit) as error:
        baseline.main()
    assert error.value.code == 2
