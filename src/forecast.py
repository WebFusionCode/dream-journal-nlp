import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO

def forecast_emotions(daily_df, periods=7):
    """Forecast sentiment/emotional trend for the next N days."""
    df = daily_df.rename(columns={"date": "ds", "sentiment": "y"}).copy()
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3))
    model.plot(forecast, ax=ax)
    ax.set_title("🪄 Forecasted Sentiment Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment")

    buf = BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    last_known = df["y"].iloc[-1]
    next_mean = forecast["yhat"].tail(periods).mean()
    change = ((next_mean - last_known) / abs(last_known + 1e-6)) * 100

    summary = (
        f"Based on your recent dreams, sentiment is predicted to "
        f"{'increase' if change > 0 else 'decrease'} by {abs(change):.1f}% "
        f"over the next {periods} days."
    )

    return buf, summary
