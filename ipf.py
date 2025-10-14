"""
Synthetic Homelessness Risk Dataset Generation using IPF
=========================================================

This script generates a synthetic dataset for homelessness risk prediction using
Iterative Proportional Fitting (IPF) to match real-world demographic patterns.

Data sources:
- Australian Bureau of Statistics (ABS) 2024
- Australian Institute of Health and Welfare (AIHW) SHS 24
"""

import numpy as np
import pandas as pd
from ipfn import ipfn
from itertools import product

import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Total populations
TOTAL_SHS_CLIENTS = 95359  # TODO - IMPLEMENT % Based System for SHS side
                           # TODO - FIND A WAY TO BEST TUNE RATIO

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# REFERENCE RATES (from research)
# ============================================================================

# Load reference rates from CSV
ref_rates = pd.read_csv('/Users/arona/Documents/GitHub/Visionary-Model/model/data/ipf_reference_rates.csv')

# ============================================================================
# SHS CLIENT DATA (June 2025)
# ============================================================================

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


def check_col_sums(df2d, gender_totals, name):
    diff = df2d.sum(axis=0) - pd.Series(gender_totals)
    if not (diff.abs() < 1e-9).all():
        print(f"[!] {name} off vs gender totals: {dict(diff)}")
        
        
# ============================================================================
# IPF IMPLEMENTATION
# ============================================================================

def create_seed_matrix(population_type='SHS'):
    """
    Create initial seed matrix with all feature combinations

    Args:
        population_type: 'SHS' or 'General'
    """

    genders = ['Male', 'Female']
    ages = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    locations = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT']
    drug_status = ['Drug', 'No_Drug']
    mental_status = ['Mental', 'Non_Mental']
    indigenous_status = ['Indigenous', 'Non_Indigenous']
    dv_status = ['DV', 'No_DV']
    homeless_status = ['Homeless', 'Not_Homeless']

    # Create all combinations
    combinations = list(product(
        genders, ages, locations, drug_status,
        mental_status, indigenous_status, dv_status, homeless_status
    ))

    df = pd.DataFrame(combinations, columns=[
        'Gender', 'Age', 'Location', 'Drug',
        'Mental', 'Indigenous', 'DV', 'Homeless'
    ])

    # Initialize with small uniform counts (seed)
    # Use different seed values to help with convergence
    if population_type == 'SHS':
        df['count'] = 1.0
    else:
        df['count'] = 5.0  # Larger seed for larger population

    df['Population'] = population_type

    return df


def run_ipf_for_population(df_seed, marginals_dict, population_name):
    """
    Run IPF algorithm for a single population

    Args:
        df_seed: Seed dataframe
        marginals_dict: Dictionary of marginal distributions
        population_name: 'SHS' or 'General'
    """

    print(f"\n{'='*80}")
    print(f"Running IPF for {population_name} Population")
    print(f"{'='*80}")
    print(f"Initial seed: {len(df_seed)} combinations")
    print(f"Initial total: {df_seed['count'].sum():.0f}")

    # Define fixed category orders for all variables
    cats = {
        'Gender': ['Male', 'Female'],
        'Age': ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'Location': ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT'],
        'Drug': ['Drug', 'No_Drug'],
        'Mental': ['Mental', 'Non_Mental'],
        'Indigenous': ['Indigenous', 'Non_Indigenous'],
        'DV': ['DV', 'No_DV'],
        'Homeless': ['Homeless', 'Not_Homeless'],
    }

    # Convert all columns to strings to avoid categorical dtype issues
    for col in ['Gender', 'Age', 'Location', 'Drug', 'Mental', 'Indigenous', 'DV', 'Homeless']:
        df_seed[col] = df_seed[col].astype(str)

    def to_series_2d(df2, row, col):
        """Build plain (non-categorical) Series with clean MultiIndex"""
        # df2: your 2D table with row index = row categories, columns = col categories
        s = (df2.reindex(index=cats[row], columns=cats[col])  # ensure full grid
                .fillna(0)
                .stack())
        s.index = pd.MultiIndex.from_product([cats[row], cats[col]], names=[row, col])
        return s.astype(float)

    # Prepare aggregates
    aggregates = []
    dimensions = []

    # NOTE: Removed standalone Gender constraint - it's redundant and causes cycling
    # when all other constraints are (variable × Gender)

    # 1. Age x Gender
    aggregates.append(to_series_2d(marginals_dict['age_gender'], 'Age', 'Gender'))
    dimensions.append(['Age', 'Gender'])

    # 2. Location x Gender
    aggregates.append(to_series_2d(marginals_dict['location_gender'], 'Location', 'Gender'))
    dimensions.append(['Location', 'Gender'])

    # 3. Drug x Gender
    aggregates.append(to_series_2d(marginals_dict['drug_gender'], 'Drug', 'Gender'))
    dimensions.append(['Drug', 'Gender'])

    # 4. Mental x Gender
    aggregates.append(to_series_2d(marginals_dict['mental_gender'], 'Mental', 'Gender'))
    dimensions.append(['Mental', 'Gender'])

    # 5. Indigenous x Gender
    aggregates.append(to_series_2d(marginals_dict['indigenous_gender'], 'Indigenous', 'Gender'))
    dimensions.append(['Indigenous', 'Gender'])

    # 6. DV x Gender
    aggregates.append(to_series_2d(marginals_dict['dv_gender'], 'DV', 'Gender'))
    dimensions.append(['DV', 'Gender'])

    # 7. Homeless x Gender (TARGET variable)
    aggregates.append(to_series_2d(marginals_dict['homeless_gender'], 'Homeless', 'Gender'))
    dimensions.append(['Homeless', 'Gender'])

    # Sanity assertions before running IPF
    print(f"\nValidating marginal constraints...")

    # Level names must match exactly
    for s, dims in zip(aggregates, dimensions):
        assert list(s.index.names) == dims, f"Index names mismatch: {s.index.names} != {dims}"

    # Target totals must all be identical
    totals = [float(s.sum()) for s in aggregates]
    assert max(totals) - min(totals) == 0, f"Target totals don't match: {totals}"

    print(f"  All checks passed. Total: {totals[0]:.0f}")

    print(f"\nRunning IPF with {len(aggregates)} marginal constraints...")

    # Run IPF with improved settings for convergence
    # - max_iteration: increased to 5000 to allow proper convergence
    # - convergence_rate: tightened to 1e-8 for better accuracy
    # - rate_tolerance: set to 0 to prevent early stopping while still far from convergence
    IPF = ipfn.ipfn(
        df_seed,
        aggregates,
        dimensions,
        weight_col='count',
        max_iteration=100,
        convergence_rate=1e-8,
        rate_tolerance=0,
        verbose=2
    )

    result = IPF.iteration()

    # Handle tuple return from verbose mode
    if isinstance(result, tuple):
        df_fitted = result[0]
        convergence_flag = result[1]

        # When verbose=2, third element is a DataFrame with convergence history
        if len(result) == 3:
            convergence_df = result[2]
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
        df_fitted = result

    print(f"\nIPF completed for {population_name}")
    print(f"Final total: {df_fitted['count'].sum():.0f}")

    # Check margin errors to verify convergence quality
    print(f"\n{'='*60}")
    print("MARGIN ERROR CHECK:")
    print(f"{'='*60}")

    def margin_err(df, features, target):
        """Calculate L1 and relative errors for a marginal"""
        got = df.groupby(features)['count'].sum()
        tgt = target.copy()
        # Align indices
        got, tgt = got.align(tgt, join='outer', fill_value=0)
        l1_err = (got - tgt).abs().sum()
        rel_err = (got - tgt).abs().div(tgt.replace(0, np.nan)).max()
        return l1_err, rel_err

    # Check all constraints
    constraint_names = [
        'Age×Gender', 'Location×Gender', 'Drug×Gender', 'Mental×Gender',
        'Indigenous×Gender', 'DV×Gender', 'Homeless×Gender'
    ]

    for feat, tgt, name in zip(dimensions, aggregates, constraint_names):
        l1, rel = margin_err(df_fitted, feat, tgt)
        print(f"{name:20s}: L1={l1:8.3f}, max rel err={rel:.3e}")

    print(f"{'='*60}\n")

    return df_fitted


def randomized_round_to_total(weights, target_total):
    """
    Randomized rounding that preserves the target total exactly

    Args:
        weights: Array of float weights
        target_total: Integer total to achieve

    Returns:
        Array of integers that sum to target_total
    """
    weights = np.array(weights, dtype=float)
    floors = np.floor(weights).astype(int)
    remainder = target_total - floors.sum()

    if remainder <= 0:
        return floors

    # Calculate fractional parts
    fracs = weights - floors
    frac_sum = fracs.sum()

    if frac_sum > 0:
        probs = fracs / frac_sum
        # Sample indices to increment
        indices = np.random.choice(len(weights), size=int(remainder), replace=False, p=probs)
        floors[indices] += 1

    return floors


def expand_to_individuals(df_fitted):
    """
    Convert fitted counts to individual records using randomized rounding
    that preserves marginal constraints
    """
    print("\nExpanding to individual records...")

    df_work = df_fitted.copy()

    # Apply randomized rounding within each (Homeless, Gender) slice
    # to preserve the most critical marginal constraint
    for homeless in ['Homeless', 'Not_Homeless']:
        for gender in ['Male', 'Female']:
            mask = (df_work['Homeless'] == homeless) & (df_work['Gender'] == gender)
            if mask.sum() == 0:
                continue

            slice_weights = df_work.loc[mask, 'count'].values
            slice_total = int(np.round(slice_weights.sum()))

            if slice_total > 0:
                rounded = randomized_round_to_total(slice_weights, slice_total)
                df_work.loc[mask, 'count'] = rounded

    # Convert to integers and filter zeros
    df_work['count'] = df_work['count'].astype(int)
    df_work = df_work[df_work['count'] > 0].copy()

    # Expand to individual records
    records = []
    for _, row in df_work.iterrows():
        count = int(row['count'])
        for _ in range(count):
            records.append({
                'Gender': row['Gender'],
                'Age': row['Age'],
                'Location': row['Location'],
                'Drug': row['Drug'],
                'Mental': row['Mental'],
                'Indigenous': row['Indigenous'],
                'DV': row['DV'],
                'Homeless': row['Homeless'],
                'Population': row['Population']
            })

    return pd.DataFrame(records)

def convert_to_binary(df):
    """
    Convert categorical variables to binary/numeric for model training
    """

    df_encoded = df.copy()

    # Gender: Male=1, Female=0
    df_encoded['Gender'] = (df_encoded['Gender'] == 'Male').astype(int)

    # Age: Convert to numeric (midpoint of range)
    age_mapping = {
        '0-17': 10,
        '18-24': 21,
        '25-34': 30,
        '35-44': 40,
        '45-54': 50,
        '55-64': 60,
        '65+': 70
    }
    df_encoded['Age'] = df_encoded['Age'].map(age_mapping)

    # Location: One-hot encode
    location_dummies = pd.get_dummies(df['Location'], prefix='Location')
    df_encoded = pd.concat([df_encoded.drop('Location', axis=1), location_dummies], axis=1)

    # Binary features
    df_encoded['Drug'] = (df_encoded['Drug'] == 'Drug').astype(int)
    df_encoded['Mental'] = (df_encoded['Mental'] == 'Mental').astype(int)
    df_encoded['Indigenous'] = (df_encoded['Indigenous'] == 'Indigenous').astype(int)
    df_encoded['DV'] = (df_encoded['DV'] == 'DV').astype(int)

    # Target variable (Homeless)
    df_encoded['Homeless'] = (df_encoded['Homeless'] == 'Homeless').astype(int)

    # Drop Population column and reorder (Homeless last)
    df_encoded = df_encoded.drop('Population', axis=1)
    cols = [c for c in df_encoded.columns if c != 'Homeless'] + ['Homeless']
    df_encoded = df_encoded[cols]

    return df_encoded


def print_statistics(df):
    """
    Print summary statistics of the generated dataset
    """

    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")

    print(f"\nTotal samples: {len(df)}")

    # Population breakdown
    if 'Population' in df.columns:
        print(f"\nPopulation Breakdown:")
        for pop in df['Population'].unique():
            n = (df['Population'] == pop).sum()
            pct = n / len(df) * 100
            print(f"  {pop}: {n:,} ({pct:.1f}%)")

    # Target variable
    if 'Homeless' in df.columns:
        if df['Homeless'].dtype == 'object':
            homeless_count = (df['Homeless'] == 'Homeless').sum()
        else:
            homeless_count = df['Homeless'].sum()
        homeless_pct = homeless_count / len(df) * 100
        print(f"\nHomeless:")
        print(f"  Yes: {homeless_count:,} ({homeless_pct:.1f}%)")
        print(f"  No: {len(df) - homeless_count:,} ({100 - homeless_pct:.1f}%)")

    # Risk factors (before encoding)
    if df['Drug'].dtype == 'object':
        print(f"\nRisk Factors:")
        print(f"  Mental Health: {(df['Mental'] == 'Mental').sum():,} ({(df['Mental'] == 'Mental').mean() * 100:.1f}%)")
        print(f"  Drug Issues: {(df['Drug'] == 'Drug').sum():,} ({(df['Drug'] == 'Drug').mean() * 100:.1f}%)")
        print(f"  Domestic Violence: {(df['DV'] == 'DV').sum():,} ({(df['DV'] == 'DV').mean() * 100:.1f}%)")
        print(f"  Indigenous: {(df['Indigenous'] == 'Indigenous').sum():,} ({(df['Indigenous'] == 'Indigenous').mean() * 100:.1f}%)")

        # Co-occurrence (Drug + Mental)
        drug_mental = ((df['Drug'] == 'Drug') & (df['Mental'] == 'Mental')).sum()
        drug_total = (df['Drug'] == 'Drug').sum()
        if drug_total > 0:
            print(f"\nCo-occurrence:")
            print(f"  Drug users with mental health: {drug_mental} ({drug_mental/drug_total*100:.1f}% of drug users)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to generate synthetic homelessness dataset
    """

    print(f"\n{'='*80}")
    print("SYNTHETIC HOMELESSNESS RISK DATASET GENERATION")
    print("Using Iterative Proportional Fitting (IPF)")
    print(f"{'='*80}")
    print(f"\nSHS Clients: {TOTAL_SHS_CLIENTS:,}")
    
    check_col_sums(shs_age_gender, shs_gender, "Age×Gender")
    check_col_sums(shs_location_gender, shs_gender, "Location×Gender")
    check_col_sums(shs_drug_gender, shs_gender, "Drug×Gender")
    check_col_sums(shs_mental_gender, shs_gender, "Mental×Gender")
    check_col_sums(shs_indigenous_gender, shs_gender, "Indigenous×Gender")
    check_col_sums(shs_homeless_gender, shs_gender, "Homeless×Gender")
    check_col_sums(shs_dv_gender, shs_gender, "DV×Gender")

    # Step 1: Create seed matrices
    print(f"\n{'='*80}")
    print("Step 1: Creating seed matrices")
    print(f"{'='*80}")

    df_shs_seed = create_seed_matrix('SHS')

    print(f"SHS seed: {len(df_shs_seed)} combinations")

    # Step 2: Prepare marginals
    print(f"\n{'='*80}")
    print("Step 2: Preparing marginal distributions")
    print(f"{'='*80}")

    shs_marginals = {
        'gender': shs_gender,
        'age_gender': shs_age_gender,
        'location_gender': shs_location_gender,
        'drug_gender': shs_drug_gender,
        'mental_gender': shs_mental_gender,
        'indigenous_gender': shs_indigenous_gender,
        'homeless_gender': shs_homeless_gender,
        'dv_gender': shs_dv_gender
    }

    print(" SHS marginals prepared")

    # Step 3: Run IPF for SHS population
    df_shs_fitted = run_ipf_for_population(df_shs_seed, shs_marginals, 'SHS')

    # Step 4: Expand to individuals
    print(f"\n{'='*80}")
    print("Step 5: Expanding to individual records")
    print(f"{'='*80}")

    df_shs_individuals = expand_to_individuals(df_shs_fitted)

    print(f" SHS individuals: {len(df_shs_individuals):,}")

    # Step 5: Create final dataset
    print(f"\n{'='*80}")
    df_adjusted = df_shs_individuals.copy()


    # Step 6: Shuffle
    print("\nShuffling dataset...")
    df_adjusted = df_adjusted.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Print statistics before encoding
    print_statistics(df_adjusted)

    # Step 7: Convert to binary
    print(f"\n{'='*80}")
    print("Step 9: Converting to binary features")
    print(f"{'='*80}")

    df_final = convert_to_binary(df_adjusted)

    # Step 8: Save
    print(f"\n{'='*80}")
    print("Step 10: Saving dataset")
    print(f"{'='*80}")

    output_file = '/Users/arona/Documents/GitHub/Visionary-Model/data/synthetic_homelessness_data.csv'
    df_final.to_csv(output_file, index=False)

    print(f"\n Dataset saved to: {output_file}")
    print(f" Shape: {df_final.shape}")
    print(f" Features ({len(df_final.columns)-1}): {list(df_final.columns[:-1])}")
    print(f" Target: {df_final.columns[-1]}")

    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples: {len(df_final):,}")
    print(f"Homeless (1): {df_final['Homeless'].sum():,} ({df_final['Homeless'].mean()*100:.1f}%)")
    print(f"Not Homeless (0): {(df_final['Homeless']==0).sum():,} ({(1-df_final['Homeless'].mean())*100:.1f}%)")

    print(f"\n{'='*80}")
    print(" GENERATION COMPLETE!")
    print(f"{'='*80}\n")

    return df_final


if __name__ == "__main__":
    df = main()
