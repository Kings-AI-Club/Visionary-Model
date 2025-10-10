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

# Your SHS data
TOTAL_GENERAL_POP = 0  # TODO - Changed from 500000
                            # Sample of general population (adjust as needed)

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
shs_age_gender = pd.DataFrame({
    'Male': {
        '0-17': 8306 + 5515,  # 0-9 + 10-17
        '18-24': 4647,
        '25-34': 4466,
        '35-44': 5423,
        '45-54': 5014,
        '55-64': 3123,
        '65+': 1869
    },
    'Female': {
        '0-17': 7820 + 6609,  # 0-9 + 10-17
        '18-24': 9650,
        '25-34': 12011,
        '35-44': 11695,
        '45-54': 7128,
        '55-64': 3527,
        '65+': 2120
    }
})

# Location x Gender
shs_location_gender = pd.DataFrame({
    'Male': {'NSW': 9348, 'VIC': 13042, 'QLD': 7932, 'WA': 2583,
             'SA': 2559, 'TAS': 1087, 'ACT': 837, 'NT': 1040},
    'Female': {'NSW': 14831, 'VIC': 20425, 'QLD': 11852, 'WA': 5315,
               'SA': 3784, 'TAS': 1265, 'ACT': 1162, 'NT': 2049}
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

# ============================================================================
# GENERAL POPULATION MARGINALS
# ============================================================================

def create_general_population_marginals():
    """
    Create marginal distributions for general (non-SHS) population
    based on Australian demographics
    """

    # Gender (roughly equal)
    gen_gender = {
        'Male': int(TOTAL_GENERAL_POP * 0.495),
        'Female': int(TOTAL_GENERAL_POP * 0.505)
    }

    # Age distribution (general population is older)
    gen_age_gender = pd.DataFrame({
        'Male': {
            '0-17': int(gen_gender['Male'] * 0.220),  # 22% under 18
            '18-24': int(gen_gender['Male'] * 0.097),
            '25-34': int(gen_gender['Male'] * 0.140),
            '35-44': int(gen_gender['Male'] * 0.133),
            '45-54': int(gen_gender['Male'] * 0.125),
            '55-64': int(gen_gender['Male'] * 0.115),
            '65+': int(gen_gender['Male'] * 0.170)
        },
        'Female': {
            '0-17': int(gen_gender['Female'] * 0.220),
            '18-24': int(gen_gender['Female'] * 0.097),
            '25-34': int(gen_gender['Female'] * 0.140),
            '35-44': int(gen_gender['Female'] * 0.133),
            '45-54': int(gen_gender['Female'] * 0.125),
            '55-64': int(gen_gender['Female'] * 0.115),
            '65+': int(gen_gender['Female'] * 0.170)
        }
    })

    # Location (population distribution by state)
    gen_location_gender = pd.DataFrame({
        'Male': {
            'NSW': int(gen_gender['Male'] * 0.320),
            'VIC': int(gen_gender['Male'] * 0.260),
            'QLD': int(gen_gender['Male'] * 0.200),
            'WA': int(gen_gender['Male'] * 0.110),
            'SA': int(gen_gender['Male'] * 0.070),
            'TAS': int(gen_gender['Male'] * 0.020),
            'ACT': int(gen_gender['Male'] * 0.017),
            'NT': int(gen_gender['Male'] * 0.010)
        },
        'Female': {
            'NSW': int(gen_gender['Female'] * 0.320),
            'VIC': int(gen_gender['Female'] * 0.260),
            'QLD': int(gen_gender['Female'] * 0.200),
            'WA': int(gen_gender['Female'] * 0.110),
            'SA': int(gen_gender['Female'] * 0.070),
            'TAS': int(gen_gender['Female'] * 0.020),
            'ACT': int(gen_gender['Female'] * 0.017),
            'NT': int(gen_gender['Female'] * 0.010)
        }
    })

    # Drug use (much lower in general population)
    # Male: 4.4%, Female: 2.1% (substance use disorder)
    gen_drug_gender = pd.DataFrame({
        'Male': {
            'Drug': int(gen_gender['Male'] * 0.044),
            'No_Drug': int(gen_gender['Male'] * 0.956)
        },
        'Female': {
            'Drug': int(gen_gender['Female'] * 0.021),
            'No_Drug': int(gen_gender['Female'] * 0.979)
        }
    })

    # Mental health (21.5% in general population)
    gen_mental_gender = pd.DataFrame({
        'Male': {
            'Mental': int(gen_gender['Male'] * 0.215),
            'Non_Mental': int(gen_gender['Male'] * 0.785)
        },
        'Female': {
            'Mental': int(gen_gender['Female'] * 0.215),
            'Non_Mental': int(gen_gender['Female'] * 0.785)
        }
    })

    # Indigenous (3.8% of general population)
    gen_indigenous_gender = pd.DataFrame({
        'Male': {
            'Indigenous': int(gen_gender['Male'] * 0.038),
            'Non_Indigenous': int(gen_gender['Male'] * 0.962)
        },
        'Female': {
            'Indigenous': int(gen_gender['Female'] * 0.038),
            'Non_Indigenous': int(gen_gender['Female'] * 0.962)
        }
    })

    # Homeless (very low in general population - use census rate of 0.48%)
    gen_homeless_gender = pd.DataFrame({
        'Male': {
            'Homeless': int(gen_gender['Male'] * 0.0048),
            'Not_Homeless': int(gen_gender['Male'] * 0.9952)
        },
        'Female': {
            'Homeless': int(gen_gender['Female'] * 0.0048),
            'Not_Homeless': int(gen_gender['Female'] * 0.9952)
        }
    })

    # DV (women 23% lifetime IPV, men ~7%)
    gen_dv_gender = pd.DataFrame({
        'Male': {
            'DV': int(gen_gender['Male'] * 0.073),
            'No_DV': int(gen_gender['Male'] * 0.927)
        },
        'Female': {
            'DV': int(gen_gender['Female'] * 0.230),
            'No_DV': int(gen_gender['Female'] * 0.770)
        }
    })

    return {
        'gender': gen_gender,
        'age_gender': gen_age_gender,
        'location_gender': gen_location_gender,
        'drug_gender': gen_drug_gender,
        'mental_gender': gen_mental_gender,
        'indigenous_gender': gen_indigenous_gender,
        'homeless_gender': gen_homeless_gender,
        'dv_gender': gen_dv_gender
    }


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
        max_iteration=1000,
        convergence_rate=1e-6,
        rate_tolerance=1e-8,
        verbose=1
    )

    result = IPF.iteration()

    # Handle tuple return from verbose mode
    if isinstance(result, tuple):
        df_fitted = result[0]
        convergence_flag = result[1]
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


def apply_conditional_adjustments(df):
    """
    Apply known conditional probabilities to better reflect real-world correlations

    Based on research:
    - 42% of SHS drug users also have mental health issues
    - 33% of SHS drug users have drug + mental + DV
    - Young people (16-24) have higher mental health rates (38.8%)
    """

    print("\nApplying conditional probability adjustments...")

    df = df.copy()
    np.random.seed(RANDOM_SEED)

    # For SHS population only
    shs_mask = df['Population'] == 'SHS'

    # 1. Drug + Mental correlation (42% overlap in SHS)
    # Currently, independence would give ~8.6% * 32% = 2.75%
    # We want 42% of drug users to have mental health issues
    drug_users = shs_mask & (df['Drug'] == 'Drug')
    drug_no_mental = drug_users & (df['Mental'] == 'Non_Mental')

    # Calculate how many need to flip
    n_drug_users = drug_users.sum()
    current_with_mental = (drug_users & (df['Mental'] == 'Mental')).sum()
    target_with_mental = int(n_drug_users * 0.42)
    n_to_flip = max(0, target_with_mental - current_with_mental)

    if n_to_flip > 0 and drug_no_mental.sum() > 0:
        flip_idx = np.random.choice(
            df[drug_no_mental].index,
            size=min(n_to_flip, drug_no_mental.sum()),
            replace=False
        )
        df.loc[flip_idx, 'Mental'] = 'Mental'
        print(f"  - Adjusted {len(flip_idx)} drug users to have mental health issues")

    # 2. Young people (18-24) higher mental health rates (38.8% vs 21.5% overall)
    young_shs = shs_mask & (df['Age'] == '18-24')
    young_no_mental = young_shs & (df['Mental'] == 'Non_Mental')

    n_young = young_shs.sum()
    current_young_mental = (young_shs & (df['Mental'] == 'Mental')).sum()
    target_young_mental = int(n_young * 0.388)
    n_to_flip_young = max(0, target_young_mental - current_young_mental)

    if n_to_flip_young > 0 and young_no_mental.sum() > 0:
        flip_idx = np.random.choice(
            df[young_no_mental].index,
            size=min(n_to_flip_young, young_no_mental.sum()),
            replace=False
        )
        df.loc[flip_idx, 'Mental'] = 'Mental'
        print(f"  - Adjusted {len(flip_idx)} young people to have mental health issues")

    # 3. Indigenous higher substance use (25% vs 18.6%)
    indigenous_shs = shs_mask & (df['Indigenous'] == 'Indigenous')
    indigenous_no_drug = indigenous_shs & (df['Drug'] == 'No_Drug')

    n_indigenous = indigenous_shs.sum()
    current_indigenous_drug = (indigenous_shs & (df['Drug'] == 'Drug')).sum()
    target_indigenous_drug = int(n_indigenous * 0.15)  # Approximate from 25% illicit to problematic
    n_to_flip_indig = max(0, target_indigenous_drug - current_indigenous_drug)

    if n_to_flip_indig > 0 and indigenous_no_drug.sum() > 0:
        flip_idx = np.random.choice(
            df[indigenous_no_drug].index,
            size=min(n_to_flip_indig, indigenous_no_drug.sum()),
            replace=False
        )
        df.loc[flip_idx, 'Drug'] = 'Drug'
        print(f"  - Adjusted {len(flip_idx)} Indigenous people to have drug issues")

    return df


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
    df_encoded['SHS_Client'] = (df_encoded['Population'] == 'SHS').astype(int)

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
    print(f"General Population Sample: {TOTAL_GENERAL_POP:,}")
    print(f"Total: {TOTAL_SHS_CLIENTS + TOTAL_GENERAL_POP:,}")

    # Step 1: Create seed matrices
    print(f"\n{'='*80}")
    print("Step 1: Creating seed matrices")
    print(f"{'='*80}")

    df_shs_seed = create_seed_matrix('SHS')
    df_gen_seed = create_seed_matrix('General')

    print(f"SHS seed: {len(df_shs_seed)} combinations")
    print(f"General seed: {len(df_gen_seed)} combinations")

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

    gen_marginals = create_general_population_marginals()

    print(" SHS marginals prepared")
    print(" General population marginals prepared")

    # Step 3: Run IPF for SHS population
    df_shs_fitted = run_ipf_for_population(df_shs_seed, shs_marginals, 'SHS')

    # Step 4: Run IPF for General population
    df_gen_fitted = run_ipf_for_population(df_gen_seed, gen_marginals, 'General')

    # Step 5: Expand to individuals
    print(f"\n{'='*80}")
    print("Step 5: Expanding to individual records")
    print(f"{'='*80}")

    df_shs_individuals = expand_to_individuals(df_shs_fitted)
    df_gen_individuals = expand_to_individuals(df_gen_fitted)

    print(f" SHS individuals: {len(df_shs_individuals):,}")
    print(f" General individuals: {len(df_gen_individuals):,}")

    # Step 6: Combine
    print(f"\n{'='*80}")
    print("Step 6: Combining populations")
    print(f"{'='*80}")

    df_combined = pd.concat([df_shs_individuals, df_gen_individuals], ignore_index=True)

    # Step 7: Apply conditional adjustments
    df_adjusted = apply_conditional_adjustments(df_combined)

    # Step 8: Shuffle
    print("\nShuffling dataset...")
    df_adjusted = df_adjusted.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Print statistics before encoding
    print_statistics(df_adjusted)

    # Step 9: Convert to binary
    print(f"\n{'='*80}")
    print("Step 9: Converting to binary features")
    print(f"{'='*80}")

    df_final = convert_to_binary(df_adjusted)

    # Step 10: Save
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
