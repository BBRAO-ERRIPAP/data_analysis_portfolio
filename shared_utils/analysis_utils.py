import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

PALETTE   = ["#2D6A4F","#52B788","#95D5B2","#D8F3DC","#1B4332","#40916C","#74C69D","#B7E4C7"]
ACCENT    = "#2D6A4F"
SECONDARY = "#52B788"

def set_global_style():
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA", "axes.facecolor": "#FAFAFA",
        "axes.edgecolor":   "#CCCCCC", "axes.grid": True,
        "grid.alpha": 0.35,            "grid.linestyle": "--",
        "font.family": "DejaVu Sans",  "font.size": 11,
        "axes.titlesize": 14,          "axes.titleweight": "bold",
        "axes.labelsize": 12,
    })
    sns.set_palette(PALETTE)

set_global_style()

# ── Data Generators ──────────────────────────────────────────────────────────

def generate_sales_data(n=5000, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", "2023-12-31", periods=n)
    cats  = ["Electronics","Clothing","Home & Garden","Sports","Books"]
    regs  = ["North","South","East","West","Central"]
    return pd.DataFrame({
        "OrderDate":  dates,
        "OrderID":    [f"ORD{i:05d}" for i in range(n)],
        "CustomerID": np.random.randint(1000, 2000, n),
        "Product":    np.random.choice(cats, n, p=[0.30,0.25,0.20,0.15,0.10]),
        "Region":     np.random.choice(regs, n),
        "Sales":      np.abs(np.random.normal(1500, 600, n)),
        "Quantity":   np.random.randint(1, 10, n),
        "Discount":   np.random.uniform(0, 0.3, n),
        "Profit":     np.abs(np.random.normal(300, 150, n)),
    })

def generate_healthcare_data(n=3000, seed=7):
    np.random.seed(seed)
    depts = ["Cardiology","Orthopedics","Pediatrics","Neurology","Oncology"]
    return pd.DataFrame({
        "PatientID":     [f"P{i:05d}" for i in range(n)],
        "Age":           np.random.randint(18, 85, n),
        "Gender":        np.random.choice(["Male","Female"], n),
        "Department":    np.random.choice(depts, n),
        "AdmissionDate": pd.date_range("2023-01-01", periods=n, freq="3h"),
        "LengthOfStay":  np.abs(np.random.normal(5,3,n)).astype(int)+1,
        "TreatmentCost": np.abs(np.random.normal(45000,20000,n)),
        "Satisfaction":  np.clip(np.random.normal(4.0,0.6,n), 1, 5),
        "Readmission":   np.random.choice([0,1], n, p=[0.917,0.083]),
        "InsuranceType": np.random.choice(["Private","Government","Self-Pay"], n),
    })

def generate_sports_data(n=500, seed=99):
    np.random.seed(seed)
    teams     = ["Team Alpha","Team Beta","Team Gamma","Team Delta",
                 "Team Epsilon","Team Zeta","Team Eta","Team Theta"]
    positions = ["Forward","Midfielder","Defender","Goalkeeper"]
    return pd.DataFrame({
        "PlayerID":    [f"PLY{i:04d}" for i in range(n)],
        "Team":        np.random.choice(teams, n),
        "Position":    np.random.choice(positions, n, p=[0.25,0.35,0.30,0.10]),
        "Age":         np.random.randint(18, 38, n),
        "Goals":       np.abs(np.random.poisson(8, n)),
        "Assists":     np.abs(np.random.poisson(5, n)),
        "Matches":     np.random.randint(10, 38, n),
        "PassAcc":     np.clip(np.random.normal(78,10,n), 50, 99),
        "Rating":      np.clip(np.random.normal(7.0,0.8,n), 5, 10),
        "MarketValue": np.abs(np.random.lognormal(15,1.5,n)),
    })

def generate_stock_data(n_days=500, seed=21):
    np.random.seed(seed)
    dates   = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = ["TECH","HEALTH","ENERGY","FINANCE","CONSUMER"]
    rows = []
    for t in tickers:
        price = 100.0
        for d in dates:
            price *= (1 + np.random.normal(0.0003, 0.015))
            rows.append({"Date":d, "Ticker":t, "Close":round(price,2),
                         "Volume": int(abs(np.random.normal(1e6, 3e5)))})
    df = pd.DataFrame(rows)
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    return df

def generate_ecommerce_data(n=8000, seed=55):
    np.random.seed(seed)
    cats    = ["Fashion","Electronics","Beauty","Home","Food"]
    devices = ["Mobile","Desktop","Tablet"]
    return pd.DataFrame({
        "CustomerID": np.random.randint(1, 1500, n),
        "OrderDate":  pd.to_datetime(
                        np.random.choice(pd.date_range("2023-01-01","2023-12-31"), n)),
        "Category":   np.random.choice(cats, n),
        "OrderValue": np.abs(np.random.normal(2200, 900, n)),
        "Quantity":   np.random.randint(1, 8, n),
        "Device":     np.random.choice(devices, n, p=[0.55,0.35,0.10]),
        "ReturnFlag": np.random.choice([0,1], n, p=[0.85,0.15]),
        "Rating":     np.random.randint(1, 6, n),
        "DeliveryDays": np.random.randint(1, 10, n),
    })

# ── Shared Functions ─────────────────────────────────────────────────────────

def eda_summary(df):
    return {
        "shape":   df.shape,
        "missing": df.isnull().sum().to_dict(),
        "nunique": {c: df[c].nunique() for c in df.columns},
        "stats":   df.describe().round(2).to_dict(),
    }

def compute_rfm(df, customer_col="CustomerID", date_col="OrderDate",
                value_col="Sales", order_col="OrderID"):
    max_date = df[date_col].max()
    rfm = df.groupby(customer_col).agg(
        Recency   = (date_col,  lambda x: (max_date - x.max()).days),
        Frequency = (order_col, "nunique"),
        Monetary  = (value_col, "sum"),
    ).reset_index()
    for col in ["Recency","Frequency","Monetary"]:
        rfm[f"{col}_Score"] = pd.qcut(
            rfm[col] if col=="Recency" else rfm[col].rank(method="first"),
            q=4, labels=[4,3,2,1] if col=="Recency" else [1,2,3,4]
        ).astype(int)
    rfm["RFM_Score"] = (rfm["Recency_Score"].astype(str)
                       + rfm["Frequency_Score"].astype(str)
                       + rfm["Monetary_Score"].astype(str))
    rfm["Segment"] = rfm["RFM_Score"].apply(_rfm_segment)
    return rfm

def _rfm_segment(score):
    r,f,m = int(score[0]),int(score[1]),int(score[2])
    if r>=4 and f>=4 and m>=4: return "Champions"
    if r>=3 and f>=3:          return "Loyal Customers"
    if r>=4 and f<=2:          return "New Customers"
    if r<=2 and f>=3:          return "At Risk"
    if r==1:                   return "Lost"
    return "Potential Loyalists"

def kmeans_clusters(df, features, n_clusters=4, seed=42):
    X  = df[features].dropna()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    df = df.copy()
    df.loc[X.index, "Cluster"] = km.fit_predict(Xs)
    print(f"  Silhouette: {silhouette_score(Xs, km.labels_):.3f}")
    return df

def save_fig(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")