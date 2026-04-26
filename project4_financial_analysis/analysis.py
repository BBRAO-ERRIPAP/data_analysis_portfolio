import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared_utils"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from analysis_utils import (generate_stock_data, save_fig, PALETTE, ACCENT, SECONDARY)

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)
RISK_FREE = 0.06 / 252

def load_data():
    df = generate_stock_data(500)
    print(f"  Rows: {len(df):,}  |  Tickers: {df['Ticker'].nunique()}")
    print(f"  Range: {df['Date'].min().date()} – {df['Date'].max().date()}")
    return df

def plot_price_trends(df):
    fig, ax = plt.subplots(figsize=(13,5))
    for i,(tick,grp) in enumerate(df.groupby("Ticker")):
        ax.plot(grp["Date"], grp["Close"], lw=1.6, color=PALETTE[i], label=tick)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.set(title="Stock Price Trends", ylabel="Price (₹)")
    ax.legend(ncol=3)
    save_fig(fig, f"{OUT}/01_price_trends.png")

def plot_returns_distribution(df):
    tickers = df["Ticker"].unique()
    fig, axes = plt.subplots(1, len(tickers), figsize=(14,4), sharey=False)
    for ax,(tick,grp) in zip(axes, df.groupby("Ticker")):
        ret = grp["Return"].dropna() * 100
        ax.hist(ret, bins=35, color=ACCENT, alpha=0.75, edgecolor="white")
        ax.axvline(ret.mean(), color="red", lw=1.5, linestyle="--",
                   label=f"μ={ret.mean():.2f}%")
        ax.set(title=tick, xlabel="Daily Return (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Daily Return Distributions", fontweight="bold")
    save_fig(fig, f"{OUT}/02_returns_distribution.png")

def plot_correlation(df):
    pivot = df.pivot(index="Date", columns="Ticker", values="Return").dropna()
    corr  = pivot.corr()
    fig, ax = plt.subplots(figsize=(7,6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, square=True,
                linewidths=0.4, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Return Correlation Matrix")
    save_fig(fig, f"{OUT}/03_correlation_matrix.png")
    return pivot

def plot_rolling_volatility(df, window=21):
    fig, ax = plt.subplots(figsize=(13,5))
    for i,(tick,grp) in enumerate(df.groupby("Ticker")):
        vol = grp.set_index("Date")["Return"].rolling(window).std() * np.sqrt(252) * 100
        ax.plot(vol.index, vol, lw=1.4, color=PALETTE[i], label=tick)
    ax.set(title=f"{window}-Day Rolling Annualised Volatility",
           ylabel="Volatility (%)")
    ax.legend(ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    save_fig(fig, f"{OUT}/04_rolling_volatility.png")

def portfolio_analysis(pivot, n_portfolios=2000):
    tickers = pivot.columns.tolist()
    rets    = pivot.dropna()
    mu      = rets.mean().values
    cov     = rets.cov().values
    np.random.seed(42)
    results = []
    for _ in range(n_portfolios):
        w     = np.random.dirichlet(np.ones(len(tickers)))
        p_ret = np.dot(w, mu) * 252 * 100
        p_vol = np.sqrt(w @ cov @ w) * np.sqrt(252) * 100
        p_shr = (p_ret - RISK_FREE*252*100) / p_vol
        results.append({"Return":p_ret, "Volatility":p_vol, "Sharpe":p_shr})
    res  = pd.DataFrame(results)
    best = res.loc[res["Sharpe"].idxmax()]
    fig, ax = plt.subplots(figsize=(9,6))
    sc = ax.scatter(res["Volatility"], res["Return"],
                    c=res["Sharpe"], cmap="RdYlGn", alpha=0.5, s=12)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.scatter(best["Volatility"], best["Return"],
               color="blue", s=100, zorder=5, marker="*",
               label=f"Best Sharpe: {best['Sharpe']:.2f}")
    ax.set(title="Efficient Frontier (Monte Carlo)",
           xlabel="Annualised Volatility (%)", ylabel="Annualised Return (%)")
    ax.legend()
    save_fig(fig, f"{OUT}/05_efficient_frontier.png")
    print(f"  Best Sharpe: {best['Sharpe']:.2f}  "
          f"Return: {best['Return']:.1f}%  Vol: {best['Volatility']:.1f}%")

if __name__ == "__main__":
    print("\n▶  Project 4: Financial Analysis")
    df    = load_data()
    plot_price_trends(df)
    plot_returns_distribution(df)
    pivot = plot_correlation(df)
    plot_rolling_volatility(df)
    portfolio_analysis(pivot)
    print("✔  Done\n")