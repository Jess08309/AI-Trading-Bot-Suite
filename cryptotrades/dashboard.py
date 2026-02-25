import os
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

PRICE_FILE = os.getenv("PRICE_HISTORY_FILE", "price_history.csv")
TRADE_FILE = os.getenv("TRADE_HISTORY_FILE", "trade_history.csv")
DEFAULT_SYMBOL = os.getenv("DASH_DEFAULT_SYMBOL", "BTC-USD")
DEFAULT_SYMBOLS = [s.strip() for s in os.getenv("DASH_DEFAULT_SYMBOLS", DEFAULT_SYMBOL).split(",") if s.strip()]

SYMBOL_OPTIONS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "ADA-USD",
    "AVAX-USD",
    "DOGE-USD",
    "MATIC-USD",
    "LTC-USD",
    "LINK-USD",
    "PI_XBTUSD",
    "PI_ETHUSD",
    "PI_SOLUSD",
    "PI_ADAUSD",
    "PI_AVAXUSD",
    "PI_DOGEUSD",
    "PI_MATICUSD",
    "PI_LTCUSD",
    "PI_LINKUSD",
    "QQQ",
    "SPY",
    "NVDA",
]

FUTURES_MAP = {
    "BTC-USD": "PI_XBTUSD",
    "ETH-USD": "PI_ETHUSD",
    "SOL-USD": "PI_SOLUSD",
    "ADA-USD": "PI_ADAUSD",
    "AVAX-USD": "PI_AVAXUSD",
    "DOGE-USD": "PI_DOGEUSD",
    "MATIC-USD": "PI_MATICUSD",
    "LTC-USD": "PI_LTCUSD",
    "LINK-USD": "PI_LINKUSD",
}


def _load_prices():
    if not os.path.exists(PRICE_FILE):
        return pd.DataFrame()
    prices = pd.read_csv(PRICE_FILE)
    if prices.empty:
        return prices
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, errors="coerce")
    prices = prices.dropna(subset=["timestamp"])
    return prices


def _load_trades():
    if not os.path.exists(TRADE_FILE):
        return pd.DataFrame()
    trades = pd.read_csv(TRADE_FILE)
    if trades.empty:
        return trades
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["timestamp"])
    return trades


def _make_candles(prices: pd.DataFrame, symbol: str, interval: str = "1min"):
    series = prices[prices["symbol"] == symbol].copy()
    if series.empty:
        return pd.DataFrame()
    series = series.set_index("timestamp").sort_index()
    ohlc = series["price"].resample(interval).ohlc().dropna()
    ohlc.reset_index(inplace=True)
    return ohlc


def _compute_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    if ohlc.empty:
        return ohlc
    df = ohlc.copy()
    close = df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace({0.0: pd.NA})
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50.0)

    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    denom = (high_14 - low_14).replace({0.0: pd.NA})
    df["stoch_k"] = ((df["close"] - low_14) / denom) * 100.0
    df["stoch_k"] = df["stoch_k"].fillna(0.0)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean().fillna(0.0)
    return df


def _format_trade_label(symbol: str, side: str, amount: float) -> str:
    if str(symbol).startswith("PI_"):
        return f"{side} {amount:.6f}"
    return f"{side} ${amount:.2f}"


def _compute_pnl_series(ohlc: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if ohlc.empty or trades.empty:
        return pd.DataFrame()
    candles = ohlc[["timestamp", "close"]].copy().sort_values("timestamp")
    t = trades.sort_values("timestamp")
    realized = 0.0
    lots = []
    rows = []
    idx = 0
    t_records = t.to_dict("records")
    for _, row in candles.iterrows():
        ts = row["timestamp"]
        price = float(row["close"])
        while idx < len(t_records) and t_records[idx]["timestamp"] <= ts:
            trade = t_records[idx]
            qty = float(trade.get("amount", 0.0))
            tprice = float(trade.get("price", 0.0))
            side = str(trade.get("side", "")).upper()
            if side == "BUY" and qty > 0:
                lots.append([qty, tprice])
            elif side == "SELL" and qty > 0:
                remaining = qty
                while remaining > 0 and lots:
                    bqty, bprice = lots[0]
                    take = min(remaining, bqty)
                    realized += take * (tprice - bprice)
                    bqty -= take
                    remaining -= take
                    if bqty <= 1e-12:
                        lots.pop(0)
                    else:
                        lots[0][0] = bqty
            idx += 1
        open_qty = sum(q for q, _ in lots)
        open_cost = sum(q * p for q, p in lots)
        avg_cost = (open_cost / open_qty) if open_qty else 0.0
        unrealized = (price - avg_cost) * open_qty if open_qty else 0.0
        rows.append({"timestamp": ts, "pnl": realized + unrealized})
    return pd.DataFrame(rows)


def _make_trade_markers(trades: pd.DataFrame, prices: pd.DataFrame, symbol: str):
    if trades.empty:
        return trades
    symbols = [symbol]
    futures_symbol = FUTURES_MAP.get(symbol)
    if futures_symbol:
        symbols.append(futures_symbol)
    subset = trades[trades["symbol"].isin(symbols)].copy()
    if subset.empty:
        return subset
    subset["plot_price"] = subset["price"]
    if futures_symbol:
        spot_prices = prices[prices["symbol"] == symbol][["timestamp", "price"]].dropna()
        if not spot_prices.empty:
            spot_prices = spot_prices.sort_values("timestamp")
            fut = subset[subset["symbol"] == futures_symbol].sort_values("timestamp")
            if not fut.empty:
                fut = pd.merge_asof(
                    fut,
                    spot_prices,
                    on="timestamp",
                    direction="nearest",
                )
                fut["plot_price"] = fut["price_y"]
                fut = fut.drop(columns=["price_y"]).rename(columns={"price_x": "price"})
                subset = pd.concat(
                    [subset[subset["symbol"] != futures_symbol], fut],
                    ignore_index=True,
                )
    return subset


def _latest_position(trades: pd.DataFrame, symbol: str):
    if trades.empty:
        return None
    sym = trades[trades["symbol"] == symbol]
    if sym.empty:
        return None
    last = sym.iloc[-1]
    try:
        balance = float(last.get("asset_balance", 0.0))
    except Exception:
        balance = 0.0
    if balance <= 0:
        return None
    entry = None
    buys = sym[sym["side"] == "BUY"]
    if not buys.empty:
        entry = float(buys.iloc[-1]["price"])
    return {
        "balance": balance,
        "entry": entry,
        "timestamp": last.get("timestamp"),
    }


app = Dash(__name__)
app.title = "CryptoTrades Live Chart"

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "12px"},
    children=[
        html.H2("CryptoTrades Live Candlestick"),
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "center"},
            children=[
                html.Label("Symbols:"),
                dcc.Dropdown(
                    id="symbols",
                    options=[{"label": s, "value": s} for s in SYMBOL_OPTIONS],
                    value=DEFAULT_SYMBOLS,
                    clearable=False,
                    multi=True,
                    style={"width": "220px"},
                ),
                html.Label("Interval:"),
                dcc.Dropdown(
                    id="interval",
                    options=[
                        {"label": "1 min", "value": "1min"},
                        {"label": "5 min", "value": "5min"},
                        {"label": "15 min", "value": "15min"},
                    ],
                    value="1min",
                    clearable=False,
                    style={"width": "140px"},
                ),
                html.Label("Panels:"),
                dcc.Checklist(
                    id="panels",
                    options=[
                        {"label": "Price", "value": "price"},
                        {"label": "MACD", "value": "macd"},
                        {"label": "RSI", "value": "rsi"},
                        {"label": "Stoch", "value": "stoch"},
                        {"label": "PnL", "value": "pnl"},
                    ],
                    value=["price"],
                    inline=True,
                    style={"display": "flex", "gap": "10px"},
                ),
            ],
        ),
        dcc.Graph(id="chart", style={"height": "75vh"}),
        dcc.Interval(id="refresh", interval=10_000, n_intervals=0),
        html.Div(id="status", style={"marginTop": "8px", "color": "#666"}),
    ],
)


@app.callback(
    Output("chart", "figure"),
    Output("status", "children"),
    Input("refresh", "n_intervals"),
    Input("symbols", "value"),
    Input("interval", "value"),
    Input("panels", "value"),
)
def update_chart(_, symbols, interval: str, panels):
    prices = _load_prices()
    trades = _load_trades()

    symbols = symbols or [DEFAULT_SYMBOL]
    panels = panels or ["price"]
    if "price" not in panels:
        panels = ["price"] + panels
    panel_order = ["price", "macd", "rsi", "stoch", "pnl"]
    panels = [p for p in panel_order if p in panels]
    rows_per_symbol = len(panels)
    total_rows = len(symbols) * rows_per_symbol
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=sum(
            ([f"{s} {p.upper()}" for p in panels] for s in symbols),
            [],
        ),
    )
    status_lines = []

    for idx, symbol in enumerate(symbols, start=1):
        base_row = (idx - 1) * rows_per_symbol + 1
        ohlc = _make_candles(prices, symbol, interval)
        if ohlc.empty:
            fig.add_annotation(
                text=f"No data for {symbol}",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=base_row,
                col=1,
            )
            status_lines.append(f"{symbol}: no price data")
            continue

        ohlc = _compute_indicators(ohlc)

        row_offsets = {panel: base_row + i for i, panel in enumerate(panels)}
        if "price" in panels:
            fig.add_trace(
                go.Candlestick(
                    x=ohlc["timestamp"],
                    open=ohlc["open"],
                    high=ohlc["high"],
                    low=ohlc["low"],
                    close=ohlc["close"],
                    name=f"{symbol} Price",
                    showlegend=False,
                ),
                row=row_offsets["price"],
                col=1,
            )

            fig.update_yaxes(title_text="Price", row=row_offsets["price"], col=1)

        trade_marks = _make_trade_markers(trades, prices, symbol)
        if "price" in panels and not trade_marks.empty:
            buys = trade_marks[trade_marks["side"] == "BUY"]
            sells = trade_marks[trade_marks["side"] == "SELL"]
            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys["timestamp"],
                        y=buys["plot_price"],
                        mode="markers",
                        hovertext=buys.apply(lambda r: _format_trade_label(r["symbol"], "BUY", r["amount"]), axis=1),
                        hoverinfo="text",
                        marker=dict(symbol="triangle-up", size=12, color="green", line=dict(color="white", width=1)),
                        name=f"{symbol} BUY",
                        showlegend=False,
                    ),
                    row=row_offsets["price"],
                    col=1,
                )
            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells["timestamp"],
                        y=sells["plot_price"],
                        mode="markers",
                        hovertext=sells.apply(lambda r: _format_trade_label(r["symbol"], "SELL", r["amount"]), axis=1),
                        hoverinfo="text",
                        marker=dict(symbol="triangle-down", size=12, color="red", line=dict(color="white", width=1)),
                        name=f"{symbol} SELL",
                        showlegend=False,
                    ),
                    row=row_offsets["price"],
                    col=1,
                )

        position = _latest_position(trades, symbol)
        if "price" in panels and position and position.get("entry"):
            fig.add_trace(
                go.Scatter(
                    x=[ohlc["timestamp"].iloc[0], ohlc["timestamp"].iloc[-1]],
                    y=[position["entry"], position["entry"]],
                    mode="lines",
                    line=dict(color="orange", dash="dash"),
                    name=f"{symbol} Entry",
                    showlegend=False,
                ),
                row=row_offsets["price"],
                col=1,
            )
            status_lines.append(
                f"{symbol}: open {position['balance']:.6f} @ {position['entry']:.2f}"
            )
        else:
            status_lines.append(f"{symbol}: flat")

        # MACD
        if "macd" in panels:
            row = row_offsets["macd"]
            fig.add_trace(
                go.Bar(
                    x=ohlc["timestamp"],
                    y=ohlc["macd_hist"],
                    marker_color=ohlc["macd_hist"].apply(lambda x: "green" if x >= 0 else "red"),
                    showlegend=False,
                    name=f"{symbol} MACD Hist",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ohlc["timestamp"],
                    y=ohlc["macd"],
                    mode="lines",
                    line=dict(color="#1f77b4"),
                    showlegend=False,
                    name=f"{symbol} MACD",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ohlc["timestamp"],
                    y=ohlc["macd_signal"],
                    mode="lines",
                    line=dict(color="#ff7f0e"),
                    showlegend=False,
                    name=f"{symbol} MACD Signal",
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text="MACD", row=row, col=1)

        # RSI
        if "rsi" in panels:
            row = row_offsets["rsi"]
            fig.add_trace(
                go.Scatter(
                    x=ohlc["timestamp"],
                    y=ohlc["rsi_14"],
                    mode="lines",
                    line=dict(color="#9467bd"),
                    showlegend=False,
                    name=f"{symbol} RSI",
                ),
                row=row,
                col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="gray", row=row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="gray", row=row, col=1)
            fig.update_yaxes(title_text="RSI", row=row, col=1, range=[0, 100])

        # Stochastic
        if "stoch" in panels:
            row = row_offsets["stoch"]
            fig.add_trace(
                go.Scatter(
                    x=ohlc["timestamp"],
                    y=ohlc["stoch_k"],
                    mode="lines",
                    line=dict(color="#2ca02c"),
                    showlegend=False,
                    name=f"{symbol} Stoch %K",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ohlc["timestamp"],
                    y=ohlc["stoch_d"],
                    mode="lines",
                    line=dict(color="#d62728"),
                    showlegend=False,
                    name=f"{symbol} Stoch %D",
                ),
                row=row,
                col=1,
            )
            fig.add_hline(y=80, line_dash="dash", line_color="gray", row=row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="gray", row=row, col=1)
            fig.update_yaxes(title_text="Stoch", row=row, col=1, range=[0, 100])

        # PnL
        if "pnl" in panels:
            row = row_offsets["pnl"]
            pnl_series = _compute_pnl_series(ohlc, trade_marks)
            if not pnl_series.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pnl_series["timestamp"],
                        y=pnl_series["pnl"],
                        mode="lines",
                        line=dict(color="black"),
                        showlegend=False,
                        name=f"{symbol} PnL",
                    ),
                    row=row,
                    col=1,
                )
            fig.update_yaxes(title_text="PnL", row=row, col=1)

        last_ts = ohlc["timestamp"].iloc[-1]
        status_lines.append(f"{symbol} last candle: {last_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    fig.update_layout(
        xaxis_title="Time",
        margin=dict(l=40, r=20, t=40, b=40),
        height=max(600, 220 * total_rows),
    )

    status = " | ".join(status_lines)
    return fig, status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
