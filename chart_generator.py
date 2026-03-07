import pandas as pd
import mplfinance as mpf

def make_candlestick_chart(symbol: str, df: pd.DataFrame, signal: str = None,
                           entry=None, take_profit=None, stop_loss=None, filename="chart.png"):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    mc = mpf.make_marketcolors(
        up="#26A69A", down="#EF5350", edge="inherit",
        wick="gray", volume="in", ohlc="inherit"
    )

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridstyle="--",
        facecolor="#0d1117",
        edgecolor="#333333",
        figcolor="#0d1117",
        rc={"font.size": 9}
    )

    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    short = df["close"].ewm(span=12).mean()
    long = df["close"].ewm(span=26).mean()
    df["MACD"] = short - long
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Hist"] = df["MACD"] - df["Signal"]

    add_plots = [
        mpf.make_addplot(df["EMA20"], color="#FFA726", width=1.0),
        mpf.make_addplot(df["EMA50"], color="#29B6F6", width=1.0),
        mpf.make_addplot(df["MACD"], panel=1, color="#29B6F6"),
        mpf.make_addplot(df["Signal"], panel=1, color="#FB8C00"),
        mpf.make_addplot(df["Hist"], panel=1, type="bar", color="#90CAF9")
    ]
    if entry:
        add_plots.append(mpf.make_addplot([entry]*len(df), color="#00E676", linestyle="--"))
    if take_profit:
        add_plots.append(mpf.make_addplot([take_profit]*len(df), color="#4CAF50", linestyle="--"))
    if stop_loss:
        add_plots.append(mpf.make_addplot([stop_loss]*len(df), color="#E53935", linestyle="--"))

    mpf.plot(
        df,
        type="candle",
        style=style,
        addplot=add_plots,
        volume=False,
        title=f"{symbol} {signal or ''}".strip(),
        ylabel="Price",
        panel_ratios=(3, 1),
        figratio=(12, 7),
        savefig=dict(fname=filename, dpi=200, bbox_inches="tight"),
    )
    return filename