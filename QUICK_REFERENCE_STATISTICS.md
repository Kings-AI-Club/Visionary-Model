# Quick Reference: Australian Population Statistics for IPF Model
**Date:** October 7, 2025

## Key Population Rates - Comparison Table

| Risk Factor | General Population | At-Risk/SHS Population | Multiplier |
|-------------|-------------------|----------------------|-----------|
| **Mental Health Issue** | 21.5% | 32.0% | 1.5x |
| **Problematic Drug Use** | 3.3%* | 8.6% | 2.6x |
| **Domestic Violence** | 23% (women lifetime) | 39% | 1.7x |
| **Indigenous** | 3.8% | 29.0% | 7.6x |

*Substance use disorder; alternative: 11.5% for any illicit drug use

## Age Distribution Comparison

| Age Group | General Pop % | SHS Clients % | Overrepresentation |
|-----------|--------------|---------------|-------------------|
| 0-14 | 16.3% | 27%* | **1.7x** |
| 15-24 | 9.7% | 19% | **2.0x** |
| 25-34 | 8.2% | 18% | **2.2x** |
| 35-44 | 7.7% | 18% | **2.3x** |
| 45-54 | 6.0% | 10% | 1.7x |
| 55+ | 22.8% | 11% | **0.5x (underrep)** |

*SHS "under 18" includes some 15-17 year olds

## Gender Patterns

| Condition | Female Rate | Male Rate | F:M Ratio |
|-----------|-------------|-----------|-----------|
| General population | 50.5% | 49.5% | 1.02:1 |
| SHS clients | 59.7% | 40.3% | 1.48:1 |
| Mental health (12-month) | Higher | Lower | ~1.6:1 |
| Anxiety disorder | 21.1% | 13.3% | 1.6:1 |
| Substance use disorder | 2.1% | 4.4% | 1:2.1 |
| Lifetime IPV | 23.0% | 7.3% | 3.2:1 |

## Critical Co-occurrence Rates (SHS Clients)

| Combination | Rate | Count |
|-------------|------|-------|
| Drug/alcohol + Mental health | 42% of drug/alcohol clients | 10,200 |
| Drug/alcohol + Mental health + DV | 33% of drug/alcohol clients | 7,900 |
| Mental health + Any other vulnerability | 52% of MH clients | 44,300 |

## Age-Specific Mental Health (General Population)

| Age Group | 12-month Disorder Rate |
|-----------|----------------------|
| 16-24 | 38.8% |
| 16-24 (Female) | 45.5% |
| 16-24 (Male) | 32.4% |
| 25-34 | 26.3% |
| 35+ | Declining with age |

## Drug Use by Age (General Population)

| Drug | Peak Age | Peak Rate |
|------|----------|-----------|
| Cannabis | 18-24 | 25.5% (12-month) |
| Cocaine | 18-24 | ~11-12% (12-month) |
| Methamphetamine | 40+ | 15.0% (lifetime) |

## Indigenous Population Characteristics

| Characteristic | Indigenous | Non-Indigenous |
|---------------|-----------|----------------|
| % of total population | 3.8% | 96.2% |
| % of SHS clients | 29.0% | 71.0% |
| Median age | 24.0 years | 38.3 years |
| Under 15 years | 33.1% | 17.9% |
| 65+ years | 5.4% | 17.2% |
| Homelessness rate | 307 per 10,000 | 35 per 10,000 |
| Any illicit drug (12-month) | 25.0% | 18.6% |

## Model Variable Mappings

### For Synthetic Data Generation (IPF)

**Binary Variables:**
- `gender`: Male (49.5% gen pop, 40% SHS), Female (50.5% gen pop, 60% SHS)
- `indigenous`: True (3.8% gen pop, 29% SHS), False
- `mental_health`: True (21.5% gen pop 16-85, 32% SHS), False
- `drug_issue`: True (3.3-8.6%), False
- `at_risk_homelessness`: True (use SHS prevalence rates), False

**Continuous Variable:**
- `age`: Use age distribution tables (see full report)

**Categorical Variables (require additional data):**
- `location`: Major city vs regional (data available in ABS reports)
- `income_source`: Government payment most common in SHS (data available)
- `education`: Correlated with age (data available)
- `employment_status`: Correlated with income source (data available)

## IPF Constraints - Recommended Setup

### Marginal Totals (proportions):

```python
# For at-risk population
marginals = {
    'gender': {'Male': 0.403, 'Female': 0.597},
    'indigenous': {True: 0.29, False: 0.71},
    'mental_health': {True: 0.32, False: 0.68},
    'drug_issue': {True: 0.086, False: 0.914},
    'age_group': {
        '0-17': 0.27,
        '18-24': 0.19,
        '25-34': 0.18,
        '35-44': 0.18,
        '45-54': 0.10,
        '55+': 0.08
    }
}

# Two-way interactions (where data available)
interactions = {
    ('mental_health', 'drug_issue'): {
        (True, True): 0.42,  # 42% of drug users have MH issue
        # Calculate others to sum to 1
    },
    ('indigenous', 'age_under_25'): {
        (True, True): 0.48,  # 48% of Indigenous SHS clients under 25
        # Calculate others
    }
}
```

## Key Data Sources (Quick Links)

1. **ABS National Study of Mental Health and Wellbeing 2020-2022**
   - Mental health prevalence by age, gender
   - Substance use disorders

2. **AIHW Specialist Homelessness Services Annual Report 2023-24**
   - SHS client characteristics
   - Risk factor prevalence in at-risk population

3. **AIHW National Drug Strategy Household Survey 2022-2023**
   - Drug use prevalence by age, gender
   - Cannabis, cocaine, methamphetamine rates

4. **ABS Regional Population by Age and Sex, June 2024**
   - General population age distribution

5. **ABS Personal Safety Survey 2021-22**
   - Domestic violence victimization rates

6. **ABS Indigenous Population Estimates, June 2021**
   - Indigenous population size and age structure

## Important Notes for Modeling

1. **Age Alignment Issue:** SHS uses "under 18" while general pop uses "0-14" and "15-19". Recommend standardizing to 5-year bands.

2. **Mental Health Definition:** General population (clinical 12-month disorder) vs SHS (current issue reported by agency). May not be perfectly comparable.

3. **Drug Use Severity:** General pop ranges from 3.3% (disorder) to 11.5% (any use). SHS is "problematic" (8.6%). Use 8.6% for at-risk population.

4. **DV Measurement:** General pop is lifetime (since age 15), SHS is current support period. SHS rate (39%) may be more relevant for current risk.

5. **Co-occurrence:** Strong correlations exist. Don't assume independence. Use actual co-occurrence rates where available.

6. **Indigenous Overrepresentation:** Massive (7.6x in SHS, 8.8x in homelessness). This should be a strong predictor in the model.

7. **Age Pattern:** Young adults (15-44) dramatically overrepresented in SHS (2-2.3x). Elderly underrepresented (0.5x).

8. **Gender Pattern:** Females overrepresented in SHS (60% vs 50.5% general pop), likely due to DV and children.

## Data Confidence Levels

- ✅ **High confidence:** Overall prevalence rates, SHS totals, age/gender distributions
- ⚠️ **Medium confidence:** Specific co-occurrence rates, age-specific DV rates
- ❌ **Low confidence:** Temporal ordering (which came first), regional variations

## Files in This Repository

1. `/Australian_Population_Statistics_2024-2025.md` - Full comprehensive report (70+ pages)
2. `/model/data/ipf_reference_rates.csv` - Machine-readable rates for IPF
3. `/QUICK_REFERENCE_STATISTICS.md` - This file

For detailed explanations, sources, and data gaps, see the full report.
