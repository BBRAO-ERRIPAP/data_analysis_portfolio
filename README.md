# 📊 Data Analysis Portfolio — Month 3: Data Science & Analysis

A professional data science portfolio containing **5 end-to-end analysis projects** across
different business domains, built with Python using pandas, matplotlib, seaborn, and scikit-learn.

---

## 👨‍💻 Portfolio Overview

| Detail | Info |
|--------|------|
| Total Projects | 5 |
| Records Analyzed | 19,000+ rows across all datasets |
| Visualizations Created | 25 charts (5 per project) |
| Domains Covered | Business, Healthcare, Sports, Finance, E-commerce |
| ML Models Used | Logistic Regression, Random Forest, K-Means, Monte Carlo |
| Tools | pandas, numpy, matplotlib, seaborn, scikit-learn |

---

## 📁 Project Structure
data_analysis_portfolio/
│
├── shared_utils/
│   └── analysis_utils.py              # Shared data generators, RFM, K-Means, styling
│
├── project1_sales_analysis/
│   ├── analysis.py                    # Sales trends, RFM segmentation, heatmaps
│   └── visualizations/
│       ├── 01_monthly_trend.png
│       ├── 02_product_performance.png
│       ├── 03_regional_heatmap.png
│       ├── 04_rfm_segments.png
│       └── 05_discount_profit.png
│
├── project2_healthcare_analysis/
│   ├── analysis.py                    # Department metrics, readmission ML model
│   └── visualizations/
│       ├── 01_department_metrics.png
│       ├── 02_demographics.png
│       ├── 03_readmission_features.png
│       └── 04_satisfaction_kde.png
│
├── project3_sports_analytics/
│   ├── analysis.py                    # Player ranking, market value prediction
│   └── visualizations/
│       ├── 01_top_players.png
│       ├── 02_team_comparison.png
│       ├── 03_position_heatmap.png
│       ├── 04_market_value_features.png
│       └── 05_player_clusters.png
│
├── project4_financial_analysis/
│   ├── analysis.py                    # Stock trends, correlation, efficient frontier
│   └── visualizations/
│       ├── 01_price_trends.png
│       ├── 02_returns_distribution.png
│       ├── 03_correlation_matrix.png
│       ├── 04_rolling_volatility.png
│       └── 05_efficient_frontier.png
│
├── project5_ecommerce_analytics/
│   ├── analysis.py                    # Cohort retention, segmentation, device analysis
│   └── visualizations/
│       ├── 01_category_device.png
│       ├── 02_monthly_trend.png
│       ├── 03_returns_delivery.png
│       ├── 04_customer_segments.png
│       └── 05_cohort_retention.png
│
├── portfolio_summary.py               # Master runner — runs all 5 projects at once
├── requirements.txt                   # All Python dependencies
└── README.md                          # This file
---

## ⚙️ Setup Instructions

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/data_analysis_portfolio.git
cd data_analysis_portfolio
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the portfolio
```bash
# Run all 5 projects at once
python portfolio_summary.py

# OR run individual projects
python project1_sales_analysis/analysis.py
python project2_healthcare_analysis/analysis.py
python project3_sports_analytics/analysis.py
python project4_financial_analysis/analysis.py
python project5_ecommerce_analytics/analysis.py
```

> Charts are automatically saved inside each project's `visualizations/` folder.

---

## 📦 Dependencies
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Projects Detail

---

### 🏪 Project 1 — Retail Sales Analysis
**File:** `project1_sales_analysis/analysis.py`
**Dataset:** 5,000 sales records (2022–2023)

**What it does:**
- Analyzes monthly revenue trends across 2 years
- Breaks down performance by product category and region
- Performs RFM (Recency, Frequency, Monetary) customer segmentation
- Builds a Region × Category revenue heatmap
- Analyzes the relationship between discount rate and profit

**Techniques Used:**
- Time series grouping with pandas `resample`
- RFM scoring with `pd.qcut`
- K-Means clustering for customer segments
- Correlation analysis (discount vs profit)

**Key Outputs:**
| Metric | Value |
|--------|-------|
| Total Revenue | ₹32M+ |
| Top Category | Electronics |
| Customer Segments | 6 (Champions to Lost) |
| Visualizations | 5 charts |

**Charts Generated:**
- `01_monthly_trend.png` — Line chart of revenue over time
- `02_product_performance.png` — Bar + Pie chart by category
- `03_regional_heatmap.png` — Heatmap of region × category revenue
- `04_rfm_segments.png` — RFM bar chart + scatter plot
- `05_discount_profit.png` — Scatter with trend line

---

### 🏥 Project 2 — Healthcare Data Analysis
**File:** `project2_healthcare_analysis/analysis.py`
**Dataset:** 3,000 patient records

**What it does:**
- Compares patient volume, cost, satisfaction, and readmission rate by department
- Analyzes treatment cost distribution across age groups
- Builds a Logistic Regression model to predict patient readmission
- Visualizes satisfaction score distribution per department

**Techniques Used:**
- Logistic Regression (scikit-learn)
- Label Encoding for categorical features
- KDE (Kernel Density Estimation) plots
- Box plots for cost distribution

**Key Outputs:**
| Metric | Value |
|--------|-------|
| Total Patients | 3,000 |
| Avg Treatment Cost | ₹44,883 |
| Readmission Rate | 8.9% |
| Avg Satisfaction | 3.99 / 5.0 |
| Model ROC-AUC | 0.40+ |

**Charts Generated:**
- `01_department_metrics.png` — 4-panel department dashboard
- `02_demographics.png` — Age group cost boxplot + gender breakdown
- `03_readmission_features.png` — ML feature importance bar chart
- `04_satisfaction_kde.png` — KDE plot by department

---

### ⚽ Project 3 — Sports Analytics
**File:** `project3_sports_analytics/analysis.py`
**Dataset:** 500 player records across 8 teams

**What it does:**
- Ranks top 10 players by combined Goals + Assists
- Compares team-level goals, assists, and average rating
- Builds a heatmap of average metrics by playing position
- Trains a Random Forest model to predict player market value
- Clusters players into 4 performance profiles using K-Means

**Techniques Used:**
- Random Forest Regressor (scikit-learn)
- K-Means clustering with Silhouette Score evaluation
- Log transformation for skewed market value target
- Grouped bar charts and heatmaps

**Key Outputs:**
| Metric | Value |
|--------|-------|
| Total Players | 500 |
| Teams | 8 |
| Avg Goals | 8.2 |
| Avg Rating | 7.07 / 10 |
| Player Clusters | 4 |

**Charts Generated:**
- `01_top_players.png` — Grouped bar chart of top 10 players
- `02_team_comparison.png` — Team goals/assists + rating comparison
- `03_position_heatmap.png` — Avg metrics by position
- `04_market_value_features.png` — Random Forest feature importance
- `05_player_clusters.png` — K-Means scatter plot

---

### 📈 Project 4 — Financial Market Analysis
**File:** `project4_financial_analysis/analysis.py`
**Dataset:** 500 trading days × 5 stock tickers (2,500 rows)

**What it does:**
- Plots price trends for 5 simulated stock tickers over 2 years
- Shows daily return distributions for each ticker
- Builds a return correlation matrix between all tickers
- Calculates 21-day rolling annualised volatility
- Runs a Monte Carlo simulation (2,000 portfolios) to find the optimal portfolio using Sharpe Ratio

**Techniques Used:**
- Monte Carlo simulation for portfolio optimisation
- Rolling window calculations with `pandas.rolling`
- Efficient Frontier visualisation
- Correlation matrix with Seaborn heatmap
- Annualised return and volatility calculations

**Key Outputs:**
| Metric | Value |
|--------|-------|
| Tickers | 5 (TECH, HEALTH, ENERGY, FINANCE, CONSUMER) |
| Trading Days | 500 |
| Portfolios Simulated | 2,000 |
| Best Sharpe Ratio | ~1.98 |
| Best Portfolio Return | ~36.6% annualised |

**Charts Generated:**
- `01_price_trends.png` — Multi-line price chart
- `02_returns_distribution.png` — Histogram per ticker
- `03_correlation_matrix.png` — Heatmap (lower triangle)
- `04_rolling_volatility.png` — Rolling volatility line chart
- `05_efficient_frontier.png` — Monte Carlo scatter with best portfolio star

---

### 🛒 Project 5 — E-commerce Analytics
**File:** `project5_ecommerce_analytics/analysis.py`
**Dataset:** 8,000 order records

**What it does:**
- Breaks down revenue by product category and order share by device
- Plots monthly revenue trend across the year
- Analyzes return rate per category and delivery time distribution
- Performs RFM-based customer segmentation
- Builds a cohort retention matrix showing how customers return month-over-month

**Techniques Used:**
- Cohort analysis using `groupby` + `transform("min")`
- RFM segmentation (shared utility)
- Device and category breakdown with pie and bar charts
- Seaborn heatmap for cohort retention matrix

**Key Outputs:**
| Metric | Value |
|--------|-------|
| Total Orders | 8,000 |
| Total Revenue | ₹70.3M |
| Avg Order Value | ₹2,192 |
| Return Rate | 14.9% |
| Top Category | Electronics |
| Top Device | Mobile (55%) |

**Charts Generated:**
- `01_category_device.png` — Revenue bar + device pie chart
- `02_monthly_trend.png` — Monthly revenue line chart
- `03_returns_delivery.png` — Return rate + delivery days bar charts
- `04_customer_segments.png` — RFM segment revenue + scatter
- `05_cohort_retention.png` — Cohort retention heatmap

---

## 🔧 Shared Utilities — `shared_utils/analysis_utils.py`

This module is imported by all 5 projects and contains:

| Function | Purpose |
|----------|---------|
| `generate_sales_data()` | Creates 5,000 retail sales records |
| `generate_healthcare_data()` | Creates 3,000 patient records |
| `generate_sports_data()` | Creates 500 player records |
| `generate_stock_data()` | Creates 500-day stock price data |
| `generate_ecommerce_data()` | Creates 8,000 e-commerce orders |
| `eda_summary(df)` | Returns shape, missing values, unique counts, stats |
| `compute_rfm(df)` | Computes Recency, Frequency, Monetary scores + segments |
| `kmeans_clusters(df, features)` | Runs K-Means and returns silhouette score |
| `save_fig(fig, path)` | Saves matplotlib figure to disk at 150 DPI |
| `set_global_style()` | Applies consistent green-themed chart styling |

---

## 🤖 Machine Learning Models Used

| Project | Model | Task | Key Metric |
|---------|-------|------|------------|
| Project 1 | K-Means Clustering | Customer segmentation | Silhouette Score |
| Project 2 | Logistic Regression | Readmission prediction | ROC-AUC |
| Project 3 | Random Forest Regressor | Market value prediction | R² Score |
| Project 3 | K-Means Clustering | Player profiling | Silhouette Score |
| Project 4 | Monte Carlo Simulation | Portfolio optimisation | Sharpe Ratio |
| Project 5 | K-Means Clustering | Customer segmentation | Silhouette Score |

---

## 📊 Analysis Questions Answered

**Project 1 — Sales:**
- Which product category generates the most revenue?
- Which region-category combination performs best?
- Who are our most valuable customers (RFM Champions)?
- Does offering higher discounts reduce profit?

**Project 2 — Healthcare:**
- Which department has the highest readmission rate?
- Does age group affect treatment cost?
- What features best predict patient readmission?
- Which department has the highest patient satisfaction?

**Project 3 — Sports:**
- Who are the top performing players across all teams?
- Which position generates the most goals on average?
- What factors most influence a player's market value?
- Can players be grouped into distinct performance clusters?

**Project 4 — Finance:**
- How do 5 different stocks trend over 2 years?
- Are stock returns correlated with each other?
- Which stocks are most volatile?
- What is the optimal portfolio allocation by Sharpe Ratio?

**Project 5 — E-commerce:**
- Which product category generates the highest revenue?
- What percentage of customers place repeat orders?
- Which device type drives the most orders?
- How does customer retention change month over month?

---

## 📚 Recommended Real Datasets (Kaggle)

To replace simulated data with real datasets, swap the `generate_*()` call
with `pd.read_csv("your_file.csv")` in each project.

| Project | Dataset | Link |
|---------|---------|------|
| Sales | Superstore Sales Dataset | kaggle.com/datasets/vivek468/superstore-dataset-final |
| Healthcare | Healthcare Analytics II | kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii |
| Sports | FIFA 22 Player Dataset | kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset |
| Finance | US Stocks & ETFs | kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs |
| E-commerce | Brazilian E-commerce (Olist) | kaggle.com/datasets/olistbr/brazilian-ecommerce |

---

## Datasets

This portfolio uses programmatically simulated datasets generated using NumPy and pandas.
No external CSV files are required — all data is created automatically when running the analysis scripts.

Datasets Overview:
- Sales Dataset: 5,000 retail transaction records (2022–2023)
- Healthcare Dataset: 3,000 patient records
- Sports Dataset: 500 player records across 8 teams
- Stock Dataset: 500 trading days × 5 tickers
- E-commerce Dataset: 8,000 order records

To use real datasets, replace the generate_*() function calls with pd.read_csv("your_file.csv")
Recommended Kaggle datasets are listed in the main README.md


## 📊 Reports

| Project | Report |
|---------|--------|
| Sales Analysis | [sales_report.pdf](reports/sales_report.pdf) |
| Healthcare Analysis | [healthcare_report.pdf](reports/healthcare_report.pdf) |
| Sports Analytics | [sports_report.pdf](reports/sports_report.pdf) |
| Financial Analysis | [finance_report.pdf](reports/finance_report.pdf) |
| E-commerce Analytics | [ecommerce_report.pdf](reports/ecommerce_report.pdf) |
| Executive Summary | [executive_summary.pdf](executive_summary.pdf) |

## 🎯 Presentation

| File | Description |
|------|-------------|
| [portfolio.pptx](presentation/portfolio.pptx) | 9-slide portfolio deck covering all 5 projects |

---

## ✅ Submission Checklist

- [x] `README.md` — full portfolio documentation
- [x] `requirements.txt` — all dependencies listed
- [x] `portfolio_summary.py` — master runner script
- [x] `shared_utils/analysis_utils.py` — reusable utility module
- [x] `project1_sales_analysis/analysis.py` — sales project code
- [x] `project2_healthcare_analysis/analysis.py` — healthcare project code
- [x] `project3_sports_analytics/analysis.py` — sports project code
- [x] `project4_financial_analysis/analysis.py` — financial project code
- [x] `project5_ecommerce_analytics/analysis.py` — e-commerce project code
- [x] 25 visualizations across all projects
- [x] datasets/ — simulated datasets generated via NumPy/pandas (no CSV needed)
- [x] reports/ — report summaries included in reports/ folder
- [x] presentation/ — portfolio presentation outline in presentation/ folder

---

## 👤 Author

**Jujjuvarapu Siva Sathvik**
- GitHub: github.com/BBRAO-ERRIPAP
- Email: sathvikj0123@example.com