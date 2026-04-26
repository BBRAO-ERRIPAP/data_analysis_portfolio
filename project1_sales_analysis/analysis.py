import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared_utils"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_utils import (
    generate_sales_data, eda_summary, compute_rfm,
    save_fig, PALETTE, ACCENT, SECONDARY
)

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

def load_data():
    df = generate_sales_data(5000)
    df["Month"]   = df["OrderDate"].dt.to_period("M")
    df["Revenue"] = df["Sales"] * df["Quantity"] * (1 - df["Discount"])
    print(f"  Rows: {len(df):,}  |  Customers: {df['CustomerID'].nunique()}")
    return df

def plot_monthly_trend(df):
    monthly = (df.groupby("Month")["Revenue"].sum()
                 .reset_index()
                 .assign(Month=lambda x: x["Month"].dt.to_timestamp()))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(monthly["Month"], monthly["Revenue"]/1e6,
            color=ACCENT, lw=2, marker="o", markersize=5)
    ax.fill_between(monthly["Month"], monthly["Revenue"]/1e6,
                    alpha=0.15, color=ACCENT)
    ax.set(title="Monthly Revenue Trend (2022–2023)", ylabel="Revenue (₹ M)")
    save_fig(fig, f"{OUT}/01_monthly_trend.png")

def plot_product_performance(df):
    cat = (df.groupby("Product")["Revenue"].sum()
             .sort_values(ascending=False).reset_index())
    cat["Share"] = cat["Revenue"] / cat["Revenue"].sum() * 100
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    axes[0].barh(cat["Product"], cat["Revenue"]/1e6, color=PALETTE[:len(cat)])
    axes[0].set(title="Revenue by Category", xlabel="Revenue (₹ M)")
    for i, v in enumerate(cat["Revenue"]/1e6):
        axes[0].text(v+0.02, i, f"{cat['Share'].iloc[i]:.1f}%", va="center", fontsize=10)
    axes[1].pie(cat["Revenue"], labels=cat["Product"],
                colors=PALETTE[:len(cat)], autopct="%1.1f%%", startangle=140)
    axes[1].set_title("Revenue Share by Category")
    save_fig(fig, f"{OUT}/02_product_performance.png")

def plot_regional_heatmap(df):
    pivot = (df.pivot_table(values="Revenue", index="Region",
                            columns="Product", aggfunc="sum")
               .div(1e6).round(2))
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Greens",
                linewidths=0.5, ax=ax, annot_kws={"size":9})
    ax.set_title("Revenue Heatmap – Region × Category (₹ M)")
    save_fig(fig, f"{OUT}/03_regional_heatmap.png")

def plot_rfm_segments(df):
    rfm = compute_rfm(df, order_col="OrderID")
    seg = rfm["Segment"].value_counts()
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    seg.plot(kind="bar", color=PALETTE[:len(seg)], ax=axes[0])
    axes[0].set(title="Customer Segments (RFM)", ylabel="Customers")
    axes[0].tick_params(axis="x", rotation=30)
    sc = axes[1].scatter(rfm["Recency"], rfm["Monetary"]/1e3,
                         c=rfm["Frequency"], cmap="Greens", alpha=0.6, s=25)
    plt.colorbar(sc, ax=axes[1], label="Frequency")
    axes[1].set(title="RFM Scatter", xlabel="Recency (days)", ylabel="Monetary (₹ K)")
    save_fig(fig, f"{OUT}/04_rfm_segments.png")
    return rfm

def plot_discount_vs_profit(df):
    fig, ax = plt.subplots(figsize=(9,5))
    ax.scatter(df["Discount"], df["Profit"], alpha=0.2, s=10, color=SECONDARY)
    z  = np.polyfit(df["Discount"], df["Profit"], 1)
    x_ = np.linspace(0, 0.3, 100)
    ax.plot(x_, np.poly1d(z)(x_), color=ACCENT, lw=2, label="Trend")
    ax.set(title="Discount Rate vs Profit", xlabel="Discount", ylabel="Profit (₹)")
    ax.legend()
    save_fig(fig, f"{OUT}/05_discount_profit.png")

if __name__ == "__main__":
    print("\n▶  Project 1: Sales Analysis")
    df  = load_data()
    plot_monthly_trend(df)
    plot_product_performance(df)
    plot_regional_heatmap(df)
    rfm = plot_rfm_segments(df)
    plot_discount_vs_profit(df)
    print(f"\n  Total Revenue : ₹{df['Revenue'].sum()/1e6:.2f}M")
    print(f"  Top Category  : {df.groupby('Product')['Revenue'].sum().idxmax()}")
    print(f"  Champions     : {(rfm['Segment']=='Champions').sum()}")
    print("✔  Done\n")