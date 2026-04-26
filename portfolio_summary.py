"""Master runner — runs all 5 projects in one command."""

import os, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))

PROJECTS = [
    ("Project 1 – Sales Analysis",       "project1_sales_analysis/analysis.py"),
    ("Project 2 – Healthcare Analysis",  "project2_healthcare_analysis/analysis.py"),
    ("Project 3 – Sports Analytics",     "project3_sports_analytics/analysis.py"),
    ("Project 4 – Financial Analysis",   "project4_financial_analysis/analysis.py"),
    ("Project 5 – E-commerce Analytics", "project5_ecommerce_analytics/analysis.py"),
]

def main():
    print("\n" + "="*55)
    print("  DATA ANALYSIS PORTFOLIO — MASTER RUNNER")
    print("="*55)
    results = {}
    start   = time.time()

    for name, path in PROJECTS:
        print(f"\n{'─'*55}\n  {name}\n{'─'*55}")
        t0 = time.time()
        try:
            full = os.path.join(BASE, path)
            sys.path.insert(0, os.path.dirname(full))
            with open(full) as f:
                exec(compile(f.read(), full, "exec"),
                     {"__name__": "__main__", "__file__": full})
            results[name] = ("✔", f"{time.time()-t0:.1f}s")
        except Exception as e:
            results[name] = ("✘", str(e)[:80])

    print("\n" + "="*55)
    print("  FINAL RESULTS")
    print("="*55)
    for name,(status,info) in results.items():
        print(f"  {status}  {name:<38} {info}")
    passed = sum(1 for s,_ in results.values() if s=="✔")
    print(f"\n  {passed}/{len(PROJECTS)} projects completed in {time.time()-start:.1f}s")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()