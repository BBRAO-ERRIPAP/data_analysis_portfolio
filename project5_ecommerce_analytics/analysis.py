import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared_utils"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_utils import (
    generate_ecommerce_data, compute_rfm, kmeans_clusters,
    save_fig, PALETTE, ACCENT, SECONDARY
)

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

def load_data():
    df = generate_ecommerce_data(8000)
    df["Month"]   = df["OrderDate"].dt.to_period("M")
    df["Revenue"] = df["OrderValue"] * df["Quantity"]
    print(f"  Rows: {len(df):,}  |  Customers: {df['CustomerID'].nunique()}")
    return df

def plot_category_device(df):
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    axes[0].bar(cat_rev.index, cat_rev.values/1e6, color=PALETTE[:len(cat_rev)])
    axes[0].set(title="Revenue by Category", ylabel="Revenue (₹ M)")
    for i,v in enumerate(cat_rev.values/1e6):
        axes[0].text(i, v*1.01, f"₹{v:.1f}M", ha="center", fontsize=9)
    dev = df.groupby("Device")["CustomerID"].count()
    axes[1].pie(dev, labels=dev.index, colors=PALETTE[:3],
                autopct="%1.1f%%", startangle=140)
    axes[1].set_title("Order Share by Device")
    save_fig(fig, f"{OUT}/01_category_device.png")

def plot_monthly_trend(df):
    monthly = (df.groupby("Month")["Revenue"].sum()
                 .reset_index()
                 .assign(Month=lambda x: x["Month"].dt.to_timestamp()))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(monthly["Month"], monthly["Revenue"]/1e6,
            color=ACCENT, lw=2, marker="o")
    ax.fill_between(monthly["Month"], monthly["Revenue"]/1e6,
                    alpha=0.15, color=ACCENT)
    ax.set(title="Monthly E-commerce Revenue", ylabel="Revenue (₹ M)")
    save_fig(fig, f"{OUT}/02_monthly_trend.png")

def plot_returns_delivery(df):
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    ret_rate = (df.groupby("Category")["ReturnFlag"].mean() * 100).round(1).sort_values(ascending=False)
    ret_rate.plot(kind="bar", color=PALETTE[:len(ret_rate)], ax=axes[0])
    axes[0].set(title="Return Rate by Category (%)", ylabel="Return Rate (%)")
    axes[0].tick_params(axis="x", rotation=30)
    df["DeliveryDays"].value_counts().sort_index().plot(
        kind="bar", color=SECONDARY, ax=axes[1])
    axes[1].set(title="Orders by Delivery Days", xlabel="Days", ylabel="Orders")
    save_fig(fig, f"{OUT}/03_returns_delivery.png")

def plot_customer_segments(df):
    rfm = compute_rfm(df, date_col="OrderDate",
                      value_col="Revenue", order_col="CustomerID")
    seg_rev = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    seg_rev.plot(kind="bar", color=PALETTE[:len(seg_rev)], ax=axes[0])
    axes[0].set(title="Revenue by Customer Segment", ylabel="Revenue (₹)")
    axes[0].tick_params(axis="x", rotation=35)
    sc = axes[1].scatter(rfm["Recency"], rfm["Monetary"]/1e3,
                         c=rfm["Frequency"], cmap="Greens", alpha=0.5, s=20)
    plt.colorbar(sc, ax=axes[1], label="Frequency")
    axes[1].set(title="Customer RFM Map",
                xlabel="Recency (days)", ylabel="Monetary (₹ K)")
    save_fig(fig, f"{OUT}/04_customer_segments.png")

def plot_cohort_retention(df):
    df2 = df.copy()
    df2["CohortMonth"] = (df2.groupby("CustomerID")["OrderDate"]
                             .transform("min").dt.to_period("M"))
    df2["OrderMonth"]  = df2["OrderDate"].dt.to_period("M")
    df2["CohortIndex"] = ((df2["OrderMonth"] - df2["CohortMonth"])
                          .apply(lambda x: x.n))
    cohort = (df2.groupby(["CohortMonth","CohortIndex"])["CustomerID"]
                 .nunique().unstack())
    cohort_pct = cohort.divide(cohort[0], axis=0).round(3) * 100
    cohort_pct = cohort_pct.iloc[:, :7]
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(cohort_pct, annot=True, fmt=".0f",
                cmap="YlGn", linewidths=0.4, ax=ax)
    ax.set(title="Customer Cohort Retention (%)",
           xlabel="Months Since First Purchase",
           ylabel="Cohort Month")
    save_fig(fig, f"{OUT}/05_cohort_retention.png")

if __name__ == "__main__":
    print("\n▶  Project 5: E-commerce Analytics")
    df = load_data()
    plot_category_device(df)
    plot_monthly_trend(df)
    plot_returns_delivery(df)
    plot_customer_segments(df)
    plot_cohort_retention(df)
    print(f"\n  Total Revenue : ₹{df['Revenue'].sum()/1e6:.2f}M")
    print(f"  Return Rate   : {df['ReturnFlag'].mean():.1%}")
    print(f"  Top Category  : {df.groupby('Category')['Revenue'].sum().idxmax()}")
    print(f"  Top Device    : {df.groupby('Device')['CustomerID'].count().idxmax()}")
    print("✔  Done\n")