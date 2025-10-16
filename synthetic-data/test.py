"""
Validation Script for Synthetic SHS Dataset
-------------------------------------------

Reads the generated CSV (synthetic_shs_clients.csv) and prints:
- Total rows
- Frequency counts and percentages for each key categorical column
- Cross-tabulations for gender vs. other attributes (sanity checks)
"""

import pandas as pd

# === 1) Load CSV ===
csv_path = "data/synthetic_shs_clients.csv"  # adjust if stored elsewhere
df = pd.read_csv(csv_path)

print("============================================================")
print(f"Loaded {len(df):,} synthetic records from: {csv_path}")
print("============================================================\n")

# === 2) Summary statistics ===
def summarize_counts(col):
    print(f"\n--- {col.upper()} DISTRIBUTION ---")
    counts = df[col].value_counts(dropna=False)
    pct = df[col].value_counts(normalize=True, dropna=False) * 100
    summary = pd.DataFrame({"count": counts, "percent": pct.round(2)})
    print(summary.to_string())
    print()

for col in ["gender", "age", "location", "drug", "mental", "indigenous", "dv", "homeless"]:
    summarize_counts(col)

# === 3) Cross-check gender splits ===
print("\n============================================================")
print("CROSS-TABULATIONS (Gender vs. key features)")
print("============================================================")

for col in ["age", "location", "drug", "mental", "indigenous", "dv", "homeless"]:
    print(f"\nGender × {col.capitalize()}:")
    ctab = pd.crosstab(df["gender"], df[col], margins=True)
    print(ctab)
    print()

# === 4) Quick sanity check on total proportion ===
print("============================================================")
print("OVERALL HOMELESSNESS RATE")
print("============================================================")
homeless_rate = (df["homeless"] == "Homeless").mean() * 100
print(f"Homeless: {homeless_rate:.2f}% of total synthetic population\n")

print("Validation complete ✅")
