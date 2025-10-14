# --- 0) Imports ---
import numpy as np, pandas as pd
from ipfn import ipfn  # package name is 'ipfn'; class is 'ipfn'

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

# --- 3) Seed (uniform) tensor & IPF constraints ---
shape = (len(G), len(A), len(L), len(D), len(M), len(I), len(V), len(H))
X0 = np.ones(shape, dtype=float)

# Target arrays in the *same* axis order as specified in 'dims' below
agg_gender            = np.array([shs_gender[g] for g in G])                        # G
agg_age_gender        = shs_age_gender.loc[A, G].values                             # A×G
agg_loc_gender        = shs_location_gender.loc[L, G].values                        # L×G
agg_drug_gender       = shs_drug_gender.loc[D, G].values                            # D×G
agg_mental_gender     = shs_mental_gender.loc[M, G].values                          # M×G
agg_indigenous_gender = shs_indigenous_gender.loc[I, G].values                      # I×G
agg_dv_gender         = shs_dv_gender.loc[V, G].values                              # V×G
agg_home_gender       = shs_homeless_gender.loc[H, G].values                        # H×G

aggregates = [
    agg_gender,            # [G]
    agg_age_gender,        # [A,G]
    agg_loc_gender,        # [L,G]
    agg_drug_gender,       # [D,G]
    agg_mental_gender,     # [M,G]
    agg_indigenous_gender, # [I,G]
    agg_dv_gender,         # [V,G]
    agg_home_gender        # [H,G]
]

# Axis mapping: 0=G,1=A,2=L,3=D,4=M,5=I,6=V,7=H
dims = [
    [0],      # G
    [1,0],    # A×G
    [2,0],    # L×G
    [3,0],    # D×G
    [4,0],    # M×G
    [5,0],    # I×G
    [6,0],    # V×G
    [7,0],    # H×G
]

# --- 4) Fit with IPF ---
ipf = ipfn.ipfn(X0, aggregates, dims, max_iteration=500, verbose=2, rate_tolerance=0, convergence_rate=1e-8)

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
    'Drug×Gender':       (['drug'],       ['gender'], shs_drug_gender),
    'Mental×Gender':     (['mental'],     ['gender'], shs_mental_gender),
    'Indigenous×Gender': (['indigenous'], ['gender'], shs_indigenous_gender),
    'DV×Gender':         (['dv'],         ['gender'], shs_dv_gender),
    'Homeless×Gender':   (['homeless'],   ['gender'], shs_homeless_gender),
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
