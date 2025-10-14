"""
Synthetic Homelessness Risk Dataset Generation using IPF
=========================================================

This script generates a synthetic dataset for homelessness risk prediction using
Iterative Proportional Fitting (IPF) to match real-world demographic patterns.

Key approach:
1. Creates TWO populations: SHS clients (at-risk) + General population (not at-risk)
2. Uses different marginal constraints for each population
3. Respects known correlations (e.g., mental health + drug use overlap)
4. Applies Australian demographic data from ABS and AIHW sources

Data sources:
- Australian Bureau of Statistics (ABS) 2024
- Australian Institute of Health and Welfare (AIHW) SHS 2023-24
- Your provided SHS client data (June 2025)
"""

import numpy as np
import pandas as pd
from ipfn import ipfn
from itertools import product

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
# YOUR SHS CLIENT DATA (June 2025)
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
        '0-17': 13821 - 184,  # 0-9 + 10-17
        '18-24': 4647 - 184,
        '25-34': 4466 - 184,
        '35-44': 5423 - 184,
        '45-54': 5014 - 184,
        '55-64': 3123 - 184,
        '65+': 1869 - 184 + 2
    },
    'Female': {
        '0-17': 14429 - 325,  # 0-9 + 10-17
        '18-24': 9650 - 325,
        '25-34': 12011 - 325 - 3, # 3 removed to balance rounding
        '35-44': 11695 - 325,
        '45-54': 7128 - 325,
        '55-64': 3527 - 325,
        '65+': 2120 - 325
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

    # Prepare aggregates
    aggregates = []
    dimensions = []

    # 1. Gender
    agg_gender = df_seed.groupby('Gender')['count'].sum()
    for gender in ['Male', 'Female']:
        agg_gender.loc[gender] = marginals_dict['gender'][gender]
    aggregates.append(agg_gender)
    dimensions.append(['Gender'])

    # 2. Age x Gender
    agg_age_gender = df_seed.groupby(['Age', 'Gender'])['count'].sum().unstack(fill_value=0)
    for age in agg_age_gender.index:
        for gender in agg_age_gender.columns:
            agg_age_gender.loc[age, gender] = marginals_dict['age_gender'].loc[age, gender]
    aggregates.append(agg_age_gender.stack())
    dimensions.append(['Age', 'Gender'])

    # 3. Location x Gender
    agg_location_gender = df_seed.groupby(['Location', 'Gender'])['count'].sum().unstack(fill_value=0)
    for loc in agg_location_gender.index:
        for gender in agg_location_gender.columns:
            agg_location_gender.loc[loc, gender] = marginals_dict['location_gender'].loc[loc, gender]
    aggregates.append(agg_location_gender.stack())
    dimensions.append(['Location', 'Gender'])

    # 4. Drug x Gender
    agg_drug_gender = df_seed.groupby(['Drug', 'Gender'])['count'].sum().unstack(fill_value=0)
    for drug in agg_drug_gender.index:
        for gender in agg_drug_gender.columns:
            agg_drug_gender.loc[drug, gender] = marginals_dict['drug_gender'].loc[drug, gender]
    aggregates.append(agg_drug_gender.stack())
    dimensions.append(['Drug', 'Gender'])

    # 5. Mental x Gender
    agg_mental_gender = df_seed.groupby(['Mental', 'Gender'])['count'].sum().unstack(fill_value=0)
    for mental in agg_mental_gender.index:
        for gender in agg_mental_gender.columns:
            agg_mental_gender.loc[mental, gender] = marginals_dict['mental_gender'].loc[mental, gender]
    aggregates.append(agg_mental_gender.stack())
    dimensions.append(['Mental', 'Gender'])

    # 6. Indigenous x Gender
    agg_indigenous_gender = df_seed.groupby(['Indigenous', 'Gender'])['count'].sum().unstack(fill_value=0)
    for indigenous in agg_indigenous_gender.index:
        for gender in agg_indigenous_gender.columns:
            agg_indigenous_gender.loc[indigenous, gender] = marginals_dict['indigenous_gender'].loc[indigenous, gender]
    aggregates.append(agg_indigenous_gender.stack())
    dimensions.append(['Indigenous', 'Gender'])

    # 7. Homeless x Gender
    agg_homeless_gender = df_seed.groupby(['Homeless', 'Gender'])['count'].sum().unstack(fill_value=0)
    for homeless in agg_homeless_gender.index:
        for gender in agg_homeless_gender.columns:
            agg_homeless_gender.loc[homeless, gender] = marginals_dict['homeless_gender'].loc[homeless, gender]
    aggregates.append(agg_homeless_gender.stack())
    dimensions.append(['Homeless', 'Gender'])

    # 8. DV x Gender
    agg_dv_gender = df_seed.groupby(['DV', 'Gender'])['count'].sum().unstack(fill_value=0)
    for dv in agg_dv_gender.index:
        for gender in agg_dv_gender.columns:
            agg_dv_gender.loc[dv, gender] = marginals_dict['dv_gender'].loc[dv, gender]
    aggregates.append(agg_dv_gender.stack())
    dimensions.append(['DV', 'Gender'])

    print(f"\nRunning IPF with {len(aggregates)} marginal constraints...")

    # Run IPF (specify weight_col='count')
    IPF = ipfn.ipfn(
        df_seed,
        aggregates,
        dimensions,
        weight_col='count',
        max_iteration=100,
        convergence_rate=1e-6,
        rate_tolerance=1e-8,
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
    print(f"\n IPF completed for {population_name}")
    print(f"Final total: {df_fitted['count'].sum():.0f}")

    return df_fitted


def expand_to_individuals(df_fitted):
    """
    Convert fitted counts to individual records
    """
    print("\nExpanding to individual records...")

    # Round and convert to integers
    df_fitted['count'] = df_fitted['count'].round().astype(int)

    # Filter out zero/negative counts
    df_fitted = df_fitted[df_fitted['count'] > 0].copy()

    # Expand
    records = []
    for idx, row in df_fitted.iterrows():
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
