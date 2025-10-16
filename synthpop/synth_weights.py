"""
Synthetic data generator using weighted probabilities (no IPF).

Produces the same schema as synthpop/ipf2.py output so that
model/ipf_model.py can train directly on the generated CSV.

Approach
- Sample covariates from sensible base rates and distributions
- Compute Homeless probability via a logistic model with domain-informed weights
- Calibrate intercept to match a target overall Homeless rate
- Save to model/data/synthetic_homelessness_data.csv

Run
  python synthpop/synth_weights.py
Optional env vars
  SEED=42 TOTAL=95359 TARGET_RATE=0.53
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def sample_age_from_band(band: str, rng: np.random.Generator) -> int:
    ranges = {
        '0-17': (0, 17),
        '18-24': (18, 24),
        '25-34': (25, 34),
        '35-44': (35, 44),
        '45-54': (45, 54),
        '55-64': (55, 64),
        '65+': (65, 90),
    }
    lo, hi = ranges[band]
    return int(rng.integers(lo, hi + 1))


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_intercept(z_no_b: np.ndarray, target_rate: float, tol: float = 1e-5, max_iter: int = 50) -> float:
    """Find intercept b such that mean(sigmoid(z_no_b + b)) ~= target_rate."""
    lo, hi = -10.0, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p = logistic(z_no_b + mid).mean()
        if p > target_rate:
            hi = mid
        else:
            lo = mid
        if abs(p - target_rate) < tol:
            return mid
    return (lo + hi) / 2


def main():
    SEED = int(os.getenv('SEED', '42'))
    TOTAL = int(os.getenv('TOTAL', '95359'))
    TARGET_RATE = float(os.getenv('TARGET_RATE', '0.533'))  # target homeless prevalence

    rng = np.random.default_rng(SEED)

    # Categories
    genders = ['Male', 'Female']
    ages = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    locations = ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']  # match output column order

    # Base distributions
    # Roughly align with SHS-like mixes; adjust as desired
    gender_p = np.array([0.389, 0.611])
    age_counts = np.array([27295, 13804, 15977, 16519, 11576, 6479, 3709], dtype=float)
    age_p = age_counts / age_counts.sum()
    loc_p = np.array([0.02, 0.35, 0.03, 0.24, 0.09, 0.02, 0.21, 0.04])
    loc_p = loc_p / loc_p.sum()

    # Binary features base rates
    p_drug = 0.14
    p_mental = 0.28
    p_indigenous = 0.30
    p_dv = 0.42

    # Sample covariates
    gender = rng.choice(genders, size=TOTAL, p=gender_p)
    age_band = rng.choice(ages, size=TOTAL, p=age_p)
    age_numeric = np.array([sample_age_from_band(b, rng) for b in age_band], dtype=int)
    location = rng.choice(locations, size=TOTAL, p=loc_p)
    drug = (rng.random(TOTAL) < p_drug).astype(int)
    mental = (rng.random(TOTAL) < p_mental).astype(int)
    indigenous = (rng.random(TOTAL) < p_indigenous).astype(int)
    dv = (rng.random(TOTAL) < p_dv).astype(int)

    # Map categorical to numeric for model contribution
    gender_num = (gender == 'Male').astype(int)  # align with ipf2 encoding (Male=1 there? ipf2 used Male->0; we'll keep small weight)

    # Age band effects (domain-informed)
    age_effect_map = {
        '0-17': -0.60,
        '18-24': 0.10,
        '25-34': 0.25,
        '35-44': 0.35,
        '45-54': 0.30,
        '55-64': 0.12,
        '65+': -0.25,
    }
    age_eff = np.array([age_effect_map[b] for b in age_band])

    # Location effects (mild)
    loc_effect_map = {
        'ACT': -0.05,
        'NSW': 0.00,
        'NT': 0.10,
        'QLD': 0.02,
        'SA': 0.03,
        'TAS': -0.02,
        'VIC': 0.15,
        'WA': 0.00,
    }
    loc_eff = np.array([loc_effect_map[x] for x in location])

    # Logistic model weights (tunable via env vars)
    # Defaults chosen to ensure sensible monotonic effects:
    #   higher DV/Drug/Mental -> higher homelessness probability
    w_gender = float(os.getenv('W_GENDER', '0.08'))       # small effect
    w_drug = float(os.getenv('W_DRUG', '1.1'))            # tuned up
    w_mental = float(os.getenv('W_MENTAL', '0.7'))        # tuned up
    w_indigenous = float(os.getenv('W_INDIG', '0.25'))
    w_dv = float(os.getenv('W_DV', '1.3'))                # tuned up

    # Linear predictor without intercept
    z_no_b = (
        w_gender * gender_num
        + w_drug * drug
        + w_mental * mental
        + w_indigenous * indigenous
        + w_dv * dv
        + age_eff
        + loc_eff
    ).astype(float)

    # Calibrate intercept to match target prevalence
    b0 = calibrate_intercept(z_no_b, TARGET_RATE)
    p_h = logistic(z_no_b + b0)
    homeless = (rng.random(TOTAL) < p_h).astype(int)

    # Build DataFrame in the same schema as ipf2 output
    df = pd.DataFrame({
        'Gender': (gender == 'Male').astype(int),  # ipf_model treats 1 as Male after conversion; consistent binary
        'Age': age_numeric,
        'Drug': drug,
        'Mental': mental,
        'Indigenous': indigenous,
        'DV': dv,
        'Homeless': homeless,
    })

    # One-hot encode locations as booleans, in the exact order used by ipf_model/ipf2
    for state in ['ACT', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']:
        df[f'Location_{state}'] = (location == state)

    # Reorder columns to match ipf2
    cols = [
        'Gender', 'Age', 'Drug', 'Mental', 'Indigenous', 'DV',
        'Location_ACT', 'Location_NSW', 'Location_NT', 'Location_QLD',
        'Location_SA', 'Location_TAS', 'Location_VIC', 'Location_WA',
        'Homeless'
    ]
    df = df[cols]

    out_path = Path('model/data/synthetic_homelessness_data.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved synthetic dataset to: {out_path} (n={len(df):,})")
    print(f"Homeless rate: {df['Homeless'].mean()*100:.2f}% (target {TARGET_RATE*100:.2f}%)")
    # Quick sanity: DV effect by gender (observed)
    for g_val, g_name in [(0, 'Female'), (1, 'Male')]:
        sub = df[df['Gender'] == g_val]
        if len(sub) > 0:
            obs_dv1 = sub.loc[sub['DV'] == 1, 'Homeless'].mean()
            obs_dv0 = sub.loc[sub['DV'] == 0, 'Homeless'].mean()
            print(f"DV sanity ({g_name}): P(H=1|DV=1)={obs_dv1:.3f} vs DV=0={obs_dv0:.3f}")


if __name__ == '__main__':
    main()
