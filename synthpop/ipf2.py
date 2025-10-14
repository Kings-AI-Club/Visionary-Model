# --- 0) Imports ---
import numpy as np, pandas as pd
from ipfn import ipfn  # package name is 'ipfn'; class is 'ipfn'
from pathlib import Path

rng = np.random.default_rng(42)

# --- 1) Use YOUR marginals exactly as you pasted (already defined) ---
# Assumes you already created:
# shs_gender (dict) and these DataFrames with indexes as shown:
# shs_age_gender, shs_location_gender, shs_drug_gender, shs_mental_gender,
# shs_indigenous_gender, shs_homeless_gender, shs_dv_gender


# Gender marginals
shs_gender = {
    'Male': 37077,
    'Female': 58282
}

# TODO - Verify distribution
# Age x Gender (converting your age brackets to our groups)

# NOTE - Removed fron age-gender by equal amount to keep total consistent with SHS clients
    # It appears SHS data was flawed
    
shs_age_gender = pd.DataFrame({
    'Male': {
        '0-17': 13358,  
        '18-24': 4491,
        '25-34': 4316,
        '35-44': 5241,
        '45-54': 4846,
        '55-64': 3018,
        '65+': 1807
    },
    'Female': {
        '0-17': 13886,  
        '18-24': 9287,
        '25-34': 11559,
        '35-44': 11255,
        '45-54': 6860,
        '55-64': 3395,
        '65+': 2040
    }
})

# Location x Gender
shs_location_gender = pd.DataFrame({
    'Male': {'NSW': 9019, 'VIC': 12584, 'QLD': 7653, 'WA': 2492,
             'SA': 2469, 'TAS': 1049, 'ACT': 808, 'NT': 1003},
    'Female': {'NSW': 14244, 'VIC': 19617, 'QLD': 11383, 'WA': 5105,
               'SA': 3634, 'TAS': 1215, 'ACT': 1116, 'NT': 1968}
})

# Drug x Gender
shs_drug_gender = pd.DataFrame({
    'Male': {'Drug': 3706, 'No_Drug': 33371},
    'Female': {'Drug': 3500, 'No_Drug': 54782}
})

# Mental Health x Gender
shs_mental_gender = pd.DataFrame({
    'Male': {'Mental': 10007, 'Non_Mental': 27070},
    'Female': {'Mental': 17933, 'Non_Mental': 40349}
})

# Indigenous x Gender
shs_indigenous_gender = pd.DataFrame({
    'Male': {'Indigenous': 10377, 'Non_Indigenous': 26700},
    'Female': {'Indigenous': 17947, 'Non_Indigenous': 40335}
})

# Homeless x Gender (this is our TARGET variable)
shs_homeless_gender = pd.DataFrame({
    'Male': {'Homeless': 21792, 'Not_Homeless': 15285},
    'Female': {'Homeless': 28784, 'Not_Homeless': 29498}
})

# Domestic Violence x Gender
shs_dv_gender = pd.DataFrame({
    'Male': {'DV': 9342, 'No_DV': 27735},
    'Female': {'DV': 28106, 'No_DV': 30176}
})

# NEW MARGINALS: Age-based cross-tabulations (balanced using RAS algorithm)
# Age x Indigenous (from INDIGENOUS.1 table, balanced to match age and indigenous totals)
shs_age_indigenous = pd.DataFrame({
    'Indigenous': {
        '0-17': 8974,
        '18-24': 4848,
        '25-34': 4946,
        '35-44': 4568,
        '45-54': 2949,
        '55-64': 1450,
        '65+': 588
    },
    'Non_Indigenous': {
        '0-17': 18270,
        '18-24': 8930,
        '25-34': 10929,
        '35-44': 11928,
        '45-54': 8757,
        '55-64': 4963,
        '65+': 3259
    }
})

# Age x Mental Health (from MH.1 table, balanced to match age and mental totals)
shs_age_mental = pd.DataFrame({
    'Mental': {
        '0-17': 2892,
        '18-24': 5628,
        '25-34': 5684,
        '35-44': 6122,
        '45-54': 4585,
        '55-64': 2225,
        '65+': 803
    },
    'Non_Mental': {
        '0-17': 24352,
        '18-24': 8150,
        '25-34': 10191,
        '35-44': 10374,
        '45-54': 7121,
        '55-64': 4188,
        '65+': 3044
    }
})

# Age x Drug (from SUB.1 table, balanced to match age and drug totals)
shs_age_drug = pd.DataFrame({
    'Drug': {
        '0-17': 580,
        '18-24': 1396,
        '25-34': 1582,
        '35-44': 1840,
        '45-54': 1278,
        '55-64': 440,
        '65+': 89
    },
    'No_Drug': {
        '0-17': 26664,
        '18-24': 12382,
        '25-34': 14293,
        '35-44': 14656,
        '45-54': 10428,
        '55-64': 5973,
        '65+': 3758
    }
})

# Age x DV (from FDV.1 table, balanced to match age and DV totals)
shs_age_dv = pd.DataFrame({
    'DV': {
        '0-17': 14111,
        '18-24': 4816,
        '25-34': 6696,
        '35-44': 6578,
        '45-54': 3483,
        '55-64': 1247,
        '65+': 515
    },
    'No_DV': {
        '0-17': 13133,
        '18-24': 8962,
        '25-34': 9179,
        '35-44': 9918,
        '45-54': 8223,
        '55-64': 5166,
        '65+': 3332
    }
})


TOTAL = sum(shs_gender.values())

# --- 2) Category order (fixes axis ordering everywhere) ---
G = ['Male','Female']
A = ['0-17','18-24','25-34','35-44','45-54','55-64','65+']
L = ['NSW','VIC','QLD','WA','SA','TAS','ACT','NT']
D = ['Drug','No_Drug']
M = ['Mental','Non_Mental']
I = ['Indigenous','Non_Indigenous']
V = ['DV','No_DV']
H = ['Homeless','Not_Homeless']

# --- 3) Seed tensor & IPF constraints ---
shape = (len(G), len(A), len(L), len(D), len(M), len(I), len(V), len(H))

# Use uniform seed with small random perturbations to avoid symmetry issues
X0 = np.ones(shape, dtype=float) * (TOTAL / np.prod(shape))
# Add small random noise to break symmetry
rng_seed = np.random.default_rng(123)
X0 *= (1 + rng_seed.uniform(-0.01, 0.01, size=shape))

# Target arrays in the *same* axis order as specified in 'dims' below
agg_gender            = np.array([shs_gender[g] for g in G])                        # G
# IMPORTANT: ipfn expects dimensions within each constraint in ascending axis order.
# For any constraint involving Gender (axis 0) and another axis, we must order as [0, X].
# Therefore, we transpose A×G and L×G and H×G to G×A, G×L, G×H respectively so they
# match dimensions [0,1], [0,2], [0,7].
agg_gender_age        = shs_age_gender.loc[A, G].T.values                            # G×A
agg_gender_loc        = shs_location_gender.loc[L, G].T.values                       # G×L
agg_drug_gender       = shs_drug_gender.loc[D, G].values                            # D×G
agg_mental_gender     = shs_mental_gender.loc[M, G].values                          # M×G
agg_indigenous_gender = shs_indigenous_gender.loc[I, G].values                      # I×G
agg_dv_gender         = shs_dv_gender.loc[V, G].values                              # V×G
agg_gender_home       = shs_homeless_gender.loc[H, G].T.values                       # G×H

# NEW: Age-based cross-tabulations
agg_age_indigenous    = shs_age_indigenous.loc[A, I].values                          # A×I
agg_age_mental        = shs_age_mental.loc[A, M].values                              # A×M
agg_age_drug          = shs_age_drug.loc[A, D].values                                # A×D
agg_age_dv            = shs_age_dv.loc[A, V].values                                  # A×V

aggregates = [
    # agg_gender,            # [G]
    agg_gender_age,        # [G,A]
    agg_gender_loc,        # [G,L]
    # REPLACED Gender-based with Age-based cross-tabs for better granularity:
    # agg_drug_gender,       # [D,G] - REPLACED by agg_age_drug
    # agg_mental_gender,     # [M,G] - REPLACED by agg_age_mental
    # agg_indigenous_gender, # [I,G] - REPLACED by agg_age_indigenous
    # agg_dv_gender,         # [V,G] - REPLACED by agg_age_dv
    agg_gender_home,       # [G,H]
    # NEW: Age cross-tabs (provide more detail than Gender cross-tabs)
    agg_age_indigenous,    # [A,I]
    agg_age_mental,        # [A,M]
    agg_age_drug,          # [A,D]
    agg_age_dv,            # [A,V]
]

# Axis mapping: 0=G,1=A,2=L,3=D,4=M,5=I,6=V,7=H
dims = [
    # [0],      # G
    [0,1],    # G×A (ascending order)
    [0,2],    # G×L (ascending order)
    # REPLACED Gender-based with Age-based cross-tabs:
    # [3,0],    # D×G - REPLACED
    # [4,0],    # M×G - REPLACED
    # [5,0],    # I×G - REPLACED
    # [6,0],    # V×G - REPLACED
    [0,7],    # G×H (ascending order)
    # NEW: Age cross-tabs
    [1,5],    # A×I
    [1,4],    # A×M
    [1,3],    # A×D
    [1,6],    # A×V
]

# --- 4) Fit with IPF ---
# Increased max_iteration with proper convergence criteria
# rate_tolerance: max acceptable difference between iterations (use None to disable)
# convergence_rate: threshold for relative convergence
ipf = ipfn.ipfn(
    X0,
    aggregates,
    dims,
    max_iteration=2000,
    verbose=2,
    convergence_rate=1e-8,
    rate_tolerance=0.0,
)

res = ipf.iteration()

# Handle tuple return from verbose mode
if isinstance(res, tuple):
    df_fitted = res[0]
    convergence_flag = res[1]

    # When verbose=2, third element is a DataFrame with convergence history
    if len(res) == 3:
        convergence_df = res[2]
        print(f"\n{'='*60}")
        print("CONVERGENCE HISTORY:")
        print(f"{'='*60}")
        print(convergence_df.to_string())
        print(f"{'='*60}\n")

    if convergence_flag == 0:
        print(f"  Warning: Maximum iterations reached without full convergence")
    else:
        print(f"  Converged successfully")
else:
    df_fitted = res
    


Xhat = res[0] if isinstance(res, tuple) else res  # <- unpack if needed
Xhat = np.asarray(Xhat, dtype=float)

# (Optional) ensure total matches your target population
scale = TOTAL / Xhat.sum()
Xhat *= scale


# --- 5) Sample micro-data (exact N, high-fidelity to marginals) ---
multi = pd.MultiIndex.from_product([G,A,L,D,M,I,V,H],
                                   names=['gender','age','location','drug',
                                          'mental','indigenous','dv','homeless'])
weights = pd.Series(Xhat.ravel(), index=multi, name='weight')
probs = (weights / weights.sum()).values

choice_idx = rng.choice(len(weights), size=TOTAL, p=probs, replace=True)
tuples = [multi[i] for i in choice_idx]
df = pd.DataFrame(tuples, columns=multi.names)

# --- 6) Quick margin checks (L1 and max relative error) ---
def check_pair(df, rows, cols, target_df):
    got = df.groupby(rows + cols).size().unstack(cols[0]).reindex(
        index=target_df.index, columns=target_df.columns, fill_value=0
    )
    diff = (got - target_df).abs()
    l1 = diff.to_numpy().sum()
    max_rel = (diff / target_df.clip(lower=1)).to_numpy().max()
    return l1, max_rel, got

checks = {
    'Age×Gender':        (['age'],        ['gender'], shs_age_gender),
    'Location×Gender':   (['location'],   ['gender'], shs_location_gender),
    'Homeless×Gender':   (['homeless'],   ['gender'], shs_homeless_gender),
    # NEW: Age-based cross-tabs
    'Age×Indigenous':    (['age'],        ['indigenous'], shs_age_indigenous),
    'Age×Mental':        (['age'],        ['mental'], shs_age_mental),
    'Age×Drug':          (['age'],        ['drug'], shs_age_drug),
    'Age×DV':            (['age'],        ['dv'], shs_age_dv),
}

print("\nMARGIN ERROR CHECK (sampled microdata vs. targets):")
for name, (r,c,tgt) in checks.items():
    l1, max_rel, _ = check_pair(df, r, c, tgt)
    print(f"{name:<18}: L1={l1:8.3f}, max rel err={max_rel: .3e}")

print("\nSample preview:")
print(df.head())

# --- 7) Export to CSV ---

def sample_age_from_band(band):
    """Sample a representative age (float) from an age bracket."""
    ranges = {
        '0-17': (0, 17),
        '18-24': (18, 24),
        '25-34': (25, 34),
        '35-44': (35, 44),
        '45-54': (45, 54),
        '55-64': (55, 64),
        '65+': (65, 90)
    }
    lo, hi = ranges[band]
    return rng.integers(lo, hi + 1)

# Binary encode categorical fields
df['Gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['Age'] = df['age'].apply(sample_age_from_band)
df['Drug'] = df['drug'].map({'Drug': 1, 'No_Drug': 0})
df['Mental'] = df['mental'].map({'Mental': 1, 'Non_Mental': 0})
df['Indigenous'] = df['indigenous'].map({'Indigenous': 1, 'Non_Indigenous': 0})
df['DV'] = df['dv'].map({'DV': 1, 'No_DV': 0})
df['Homeless'] = df['homeless'].map({'Homeless': 1, 'Not_Homeless': 0})

# One-hot encode location (ACT, NSW, NT, QLD, SA, TAS, VIC, WA)
location_cols = [f'Location_{state}' for state in L]
for state in L:
    df[f'Location_{state}'] = (df['location'] == state)

# Select & reorder columns
cols = [
    'Gender', 'Age', 'Drug', 'Mental', 'Indigenous', 'DV',
    'Location_ACT', 'Location_NSW', 'Location_NT', 'Location_QLD',
    'Location_SA', 'Location_TAS', 'Location_VIC', 'Location_WA',
    'Homeless'
]
df_out = df[cols]

# Save to CSV
output_path = Path("synthetic_homelessness.csv")
df_out.to_csv(output_path, index=False)
print(f"\n✅ Synthetic dataset saved to: {output_path.resolve()}")
print(df_out.head())
