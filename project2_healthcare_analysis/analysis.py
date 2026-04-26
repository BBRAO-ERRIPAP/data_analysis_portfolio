import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared_utils"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from analysis_utils import (generate_healthcare_data, save_fig, PALETTE, ACCENT, SECONDARY)

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

def load_data():
    df = generate_healthcare_data(3000)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0,30,50,65,100],
                            labels=["18–30","31–50","51–65","65+"])
    print(f"  Patients: {len(df):,}  |  Departments: {df['Department'].nunique()}")
    return df

def plot_department_metrics(df):
    dept = (df.groupby("Department")
              .agg(Patients    =("PatientID","count"),
                   AvgCost     =("TreatmentCost","mean"),
                   AvgSatisf   =("Satisfaction","mean"),
                   ReadmitRate =("Readmission","mean"))
              .reset_index().sort_values("Patients", ascending=False))
    fig, axes = plt.subplots(2,2, figsize=(13,9))
    fig.suptitle("Department Performance Dashboard", fontsize=15, fontweight="bold")
    for ax, col, title in zip(axes.flat,
        ["Patients","AvgCost","AvgSatisf","ReadmitRate"],
        ["Patient Volume","Avg Cost (₹)","Avg Satisfaction","Readmission Rate"]):
        ax.bar(dept["Department"], dept[col], color=PALETTE[:len(dept)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    save_fig(fig, f"{OUT}/01_department_metrics.png")

def plot_demographics(df):
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    age_data = [df[df["AgeGroup"]==g]["TreatmentCost"].values/1000
                for g in ["18–30","31–50","51–65","65+"]]
    bp = axes[0].boxplot(age_data, labels=["18–30","31–50","51–65","65+"],
                         patch_artist=True)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set(title="Treatment Cost by Age Group", ylabel="Cost (₹ K)")
    gender_dept = df.groupby(["Department","Gender"])["PatientID"].count().unstack()
    gender_dept.plot(kind="bar", ax=axes[1], color=[ACCENT, SECONDARY])
    axes[1].set(title="Patients by Dept & Gender", ylabel="Count")
    axes[1].tick_params(axis="x", rotation=30)
    save_fig(fig, f"{OUT}/02_demographics.png")

def readmission_model(df):
    le  = LabelEncoder()
    df2 = df.copy()
    for col in ["Department","Gender","InsuranceType"]:
        df2[col] = le.fit_transform(df2[col])
    df2["AgeGroup"] = le.fit_transform(df2["AgeGroup"].astype(str))
    features = ["Age","LengthOfStay","TreatmentCost","Satisfaction","Department","InsuranceType"]
    X = df2[features].fillna(0)
    y = df2["Readmission"]
    X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:,1]
    print(classification_report(y_te, preds, target_names=["No","Yes"]))
    print(f"  ROC-AUC: {roc_auc_score(y_te, probs):.3f}")
    imp = pd.Series(np.abs(model.coef_[0]), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8,4))
    imp.plot(kind="barh", color=ACCENT, ax=ax)
    ax.set(title="Readmission – Feature Importance", xlabel="|Coefficient|")
    save_fig(fig, f"{OUT}/03_readmission_features.png")

def plot_satisfaction(df):
    fig, ax = plt.subplots(figsize=(9,4))
    for dept, grp in df.groupby("Department"):
        grp["Satisfaction"].plot(kind="kde", ax=ax, label=dept, lw=1.8)
    ax.set(title="Satisfaction Distribution by Department", xlabel="Score (1–5)")
    ax.legend()
    save_fig(fig, f"{OUT}/04_satisfaction_kde.png")

if __name__ == "__main__":
    print("\n▶  Project 2: Healthcare Analysis")
    df = load_data()
    plot_department_metrics(df)
    plot_demographics(df)
    readmission_model(df)
    plot_satisfaction(df)
    print(f"\n  Avg Cost       : ₹{df['TreatmentCost'].mean():,.0f}")
    print(f"  Readmit Rate   : {df['Readmission'].mean():.1%}")
    print(f"  Avg Satisfaction: {df['Satisfaction'].mean():.2f}/5.0")
    print("✔  Done\n")