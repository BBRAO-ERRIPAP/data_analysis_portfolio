import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared_utils"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from analysis_utils import (
    generate_sports_data, kmeans_clusters,
    save_fig, PALETTE, ACCENT, SECONDARY
)

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

def load_data():
    df = generate_sports_data(500)
    df["GoalsPerMatch"]   = df["Goals"]   / df["Matches"]
    df["AssistsPerMatch"] = df["Assists"] / df["Matches"]
    df["G_A_Total"]       = df["Goals"]   + df["Assists"]
    print(f"  Players: {len(df)}  |  Teams: {df['Team'].nunique()}")
    return df

def plot_top_players(df):
    top10 = df.nlargest(10, "G_A_Total")
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(top10))
    ax.bar(x-0.2, top10["Goals"],   width=0.4, label="Goals",   color=ACCENT)
    ax.bar(x+0.2, top10["Assists"], width=0.4, label="Assists", color=SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n({t})" for p,t in
                        zip(top10["PlayerID"], top10["Team"])],
                       rotation=35, ha="right", fontsize=8)
    ax.set(title="Top 10 Players by Goals + Assists", ylabel="Count")
    ax.legend()
    save_fig(fig, f"{OUT}/01_top_players.png")

def plot_team_comparison(df):
    team = (df.groupby("Team")
              .agg(Goals  =("Goals","sum"),
                   Assists=("Assists","sum"),
                   Rating =("Rating","mean"))
              .reset_index().sort_values("Rating", ascending=False))
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    team.plot(kind="bar", x="Team", y=["Goals","Assists"],
              ax=axes[0], color=[ACCENT,SECONDARY])
    axes[0].set(title="Team Goals & Assists", ylabel="Count")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].barh(team["Team"], team["Rating"], color=PALETTE[:len(team)])
    axes[1].set(title="Team Avg Player Rating", xlabel="Avg Rating")
    save_fig(fig, f"{OUT}/02_team_comparison.png")

def plot_position_heatmap(df):
    pos = (df.groupby("Position")
             .agg(Goals  =("Goals","mean"),
                  Assists=("Assists","mean"),
                  PassAcc=("PassAcc","mean"),
                  Rating =("Rating","mean"))
             .round(2))
    fig, ax = plt.subplots(figsize=(9,4))
    sns.heatmap(pos, annot=True, fmt=".2f", cmap="YlGn",
                linewidths=0.5, ax=ax)
    ax.set_title("Avg Metrics by Position")
    save_fig(fig, f"{OUT}/03_position_heatmap.png")

def market_value_model(df):
    features = ["Age","Goals","Assists","Matches","PassAcc","Rating"]
    df2 = df.dropna(subset=features+["MarketValue"])
    X = df2[features]
    y = np.log1p(df2["MarketValue"])
    X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)
    print(f"  R²: {r2_score(y_te,preds):.3f}  |  MAE: ₹{mean_absolute_error(np.expm1(y_te),np.expm1(preds)):,.0f}")
    imp = pd.Series(rf.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8,4))
    imp.plot(kind="barh", color=ACCENT, ax=ax)
    ax.set(title="Market Value – Feature Importances", xlabel="Importance")
    save_fig(fig, f"{OUT}/04_market_value_features.png")

def plot_player_clusters(df):
    df_cl = kmeans_clusters(df, ["Goals","Assists","PassAcc","Rating"], n_clusters=4)
    df_cl["Cluster"] = df_cl["Cluster"].astype(int)
    fig, ax = plt.subplots(figsize=(9,5))
    for cl in sorted(df_cl["Cluster"].unique()):
        sub = df_cl[df_cl["Cluster"]==cl]
        ax.scatter(sub["Goals"], sub["Rating"],
                   label=f"Cluster {cl}", alpha=0.6, s=25, color=PALETTE[cl])
    ax.set(title="Player Clusters: Goals vs Rating", xlabel="Goals", ylabel="Rating")
    ax.legend()
    save_fig(fig, f"{OUT}/05_player_clusters.png")

if __name__ == "__main__":
    print("\n▶  Project 3: Sports Analytics")
    df = load_data()
    plot_top_players(df)
    plot_team_comparison(df)
    plot_position_heatmap(df)
    market_value_model(df)
    plot_player_clusters(df)
    top = df.loc[df["G_A_Total"].idxmax()]
    print(f"\n  Top Player : {top['PlayerID']} ({top['Team']}) — {top['G_A_Total']:.0f} G+A")
    print(f"  Avg Rating : {df['Rating'].mean():.2f}/10")
    print("✔  Done\n")