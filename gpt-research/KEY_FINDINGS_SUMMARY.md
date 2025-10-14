# Key Findings Summary: Australian Homelessness Risk Factors 2024-2025

## Executive Summary

This research compiled comprehensive statistics from authoritative Australian sources (ABS, AIHW) covering mental health, substance use, domestic violence, and Indigenous status as they relate to homelessness risk. The data reveals stark differences between the general population and those accessing homelessness services.

---

## Top 10 Critical Findings

### 1. Indigenous Australians Face Extreme Homelessness Risk
- **7.6x overrepresented** in homelessness services (29% of SHS clients vs 3.8% of population)
- **8.8x higher** homelessness rate (307 vs 35 per 10,000)
- **11x higher** rate of using homelessness services
- Indigenous population is significantly younger (median age 24 vs 38.3)

### 2. Young People Dramatically Overrepresented
- Ages 15-44 are **2.0-2.3x overrepresented** in homelessness services
- 46% of SHS clients are under 25 (vs 26% of general population)
- Peak adult SHS age: 35-44 years (18% of SHS vs 7.7% general pop)
- Elderly (65+) dramatically **underrepresented**: 5% of SHS vs 17% general pop

### 3. Mental Health Issues Are Prevalent and Rising
- **General population:** 21.5% have 12-month mental disorder
- **SHS clients:** 32% have current mental health issue (**1.5x higher**)
- **Ever homeless:** 39.1% have mental disorder (**2x** never homeless: 19.5%)
- **Youth crisis:** 45.5% of females aged 16-24 have mental disorder (up from 15% in 2017)

### 4. Mental Health Shows Strong Age and Gender Patterns
- **Age pattern:** 38.8% (16-24) → 26.3% (25-34) → declining with age
- **Gender gap (anxiety):** Females 21.1% vs Males 13.3% (1.6:1 ratio)
- **Gender reversal (substance use):** Males 4.4% vs Females 2.1% (2.1:1 ratio)
- Young women have highest mental health burden: 45.5% prevalence

### 5. Substance Use in At-Risk Populations Much Higher
- **General population:** 3.3% substance use disorder, 11.5% any illicit drug
- **SHS clients:** 8.6% problematic drug/alcohol use (**2.6x disorder rate**)
- **Hard drug use in homeless:** Heroin 113x higher, Amphetamines 20x higher
- **Indigenous rates:** 1.3x overall illicit use, 2.3x methamphetamine use

### 6. Strong Co-Occurrence of Mental Health and Substance Use
- **42% of SHS clients** with drug/alcohol issues also have mental health issues
- **50-78% of AOD treatment clients** have comorbid mental disorder
- **33% of SHS drug/alcohol clients** have all three: mental health + substance + DV
- Dual diagnosis has worse outcomes than either condition alone

### 7. Domestic Violence Major Driver of Homelessness
- **39% of SHS clients** have experienced family/domestic violence
- **26% cite FDV as main reason** for seeking homelessness services
- **Women 3.2x more likely** than men to experience intimate partner violence
- FDV clients in SHS increased from 34% (2011-12) to 39% (2023-24)

### 8. DV, Mental Health, and Substance Use Form Vicious Cycle
- DV survivors have **2-8x higher** rates of pre-existing mental illness
- DV experience **exacerbates** existing mental illness and substance abuse
- Strong associations with PTSD, depression, chronic substance use, suicidality
- **1/3 had substance problems before homelessness, 2/3 developed them after**

### 9. Indigenous Communities Face Multiple Compounding Risk Factors
- **25% higher illicit drug use** than non-Indigenous Australians
- Drug burden **3.7x higher** for Indigenous people
- **40% of Indigenous with mental health** also use substances
- **25-40% of DV homicide victims** are Indigenous (vs 3.8% of population)
- 65% of health gap explained by social determinants + health risk factors

### 10. Women Overrepresented in Homelessness Services
- **60% of SHS clients are female** (vs 50.5% of general population)
- Likely driven by DV (81% of female DV homicide victims killed by intimate partner)
- Young women (18-34) have highest DV victimization rates
- Women with children form large proportion of homeless families

---

## Risk Factor Comparison: General vs At-Risk Population

```
Risk Factor              Gen Pop    SHS Pop    Multiplier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Indigenous status        3.8%       29.0%      7.6x
Age 35-44                7.7%       18.0%      2.3x
Mental health issue      21.5%      32.0%      1.5x
Problematic drug use     3.3%       8.6%       2.6x
DV experience (women)    23.0%      39.0%      1.7x
Multiple risk factors    Rare       Very common Many x
```

---

## Critical Interactions for IPF Modeling

### Strong Positive Correlations (use actual data, don't assume independence):

1. **Mental Health ↔ Substance Use**
   - 42% co-occurrence in SHS drug/alcohol clients
   - 50-78% co-occurrence in AOD treatment

2. **Domestic Violence ↔ Mental Health**
   - 2-8x higher MH rates among DV victims
   - Bidirectional: MH increases DV risk, DV worsens MH

3. **Domestic Violence ↔ Homelessness**
   - 39% of SHS clients experienced DV
   - DV is single largest reason for seeking SHS (26%)

4. **Indigenous Status ↔ All Risk Factors**
   - Higher rates of mental health issues
   - 1.3-2.3x higher substance use
   - Overrepresented in DV homicides
   - 7.6x overrepresentation in SHS

5. **Young Age ↔ Mental Health**
   - 38.8% of 16-24 have disorder (vs 21.5% overall)
   - Particularly high in young women (45.5%)

6. **Young Age ↔ Substance Use**
   - Cannabis: 25.5% of 18-24 (vs 11.5% overall)
   - Cocaine: 11-12% of 18-24 (vs 4.5% overall)

### Important for Model:

**High-Risk Profiles:**
1. Young Indigenous woman with mental health + DV experience
2. Male 25-44 with substance use + mental health
3. Young person (15-24) with any risk factors
4. Indigenous person with multiple risk factors
5. Woman fleeing DV with children

**Lower-Risk Profiles:**
1. Elderly (65+) without risk factors
2. Middle-aged employed person without risk factors
3. Higher education + stable employment

---

## Implications for IPF (Iterative Proportional Fitting)

### 1. Do NOT Assume Independence
Many risk factors are strongly correlated. Use multi-way tables where data exists:
- Mental health × Substance use: Use 42% co-occurrence
- Indigenous × Age: Use younger age distribution
- Gender × Mental health: Use 1.6:1 female:male for anxiety

### 2. Key Marginal Constraints

**For At-Risk Population Modeling:**
```python
{
    'indigenous': 0.29,           # 7.6x overrepresented
    'age_15_44': 0.55,            # 2+ x overrepresented
    'female': 0.60,               # 1.2x overrepresented
    'mental_health': 0.32,        # 1.5x general pop
    'drug_issue': 0.086,          # 2.6x disorder rate
    'dv_experience': 0.39,        # Female-driven
}
```

### 3. Critical Interaction Terms

Must include:
- Indigenous × Age (younger distribution)
- Mental health × Drug use (42% overlap)
- Gender × DV (3:1 female:male)
- Age × Mental health (younger = higher)
- Indigenous × All risk factors (elevated)

### 4. Conditional Probabilities to Use

```
P(Mental health | Age 16-24) = 0.388
P(Mental health | Age 16-24, Female) = 0.455
P(Mental health | Age 16-24, Male) = 0.324
P(Drug use | Mental health, SHS) = 0.42
P(Mental health | Ever homeless) = 0.391
P(Indigenous | SHS) = 0.29
P(Age < 25 | Indigenous, SHS) = 0.48
```

### 5. Avoid These Mistakes

❌ Don't use general population rates for at-risk population
❌ Don't assume mental health and substance use are independent
❌ Don't ignore the massive Indigenous overrepresentation
❌ Don't use equal age distribution (young people dominate SHS)
❌ Don't assume 50/50 gender split (60% female in SHS)

✅ Do use SHS-specific rates for at-risk modeling
✅ Do model strong correlations between risk factors
✅ Do weight Indigenous status heavily (7.6x factor)
✅ Do overweight young adults (2-2.3x factor)
✅ Do use 60/40 female/male split for SHS population

---

## Data Quality and Confidence

### High Confidence ✅
- Overall prevalence rates (large samples, authoritative sources)
- SHS client counts and demographics (full service census)
- Age and gender distributions (comprehensive data)
- Indigenous overrepresentation (consistent across multiple measures)

### Medium Confidence ⚠️
- Specific co-occurrence rates (some measured, some inferred)
- Age-specific rates beyond what's published
- Temporal relationships (which came first)
- Regional variations (mostly national averages)

### Known Limitations ❌
- Mental health definition varies (clinical vs agency-reported)
- DV measurement timeframes differ (lifetime vs current)
- Most recent mental health data is 2020-22
- Drug use likely underreported (self-report bias)
- Homeless population may be undercounted (most marginalized miss census)

---

## Recommended Model Approach

### For Generating Synthetic At-Risk Population:

1. **Start with strong marginals:**
   - Indigenous: 29%
   - Age 15-44: 55%
   - Female: 60%
   - Mental health: 32%
   - Drug issue: 8.6%
   - DV: 39%

2. **Apply conditional relationships:**
   - If Indigenous → higher drug use, younger age
   - If Age 16-24 → higher mental health (38.8%)
   - If Age 16-24 + Female → very high mental health (45.5%)
   - If Drug issue → 42% also have mental health
   - If Female → higher DV (3:1 ratio)

3. **Use IPF to balance constraints:**
   - Fit to marginals while preserving correlations
   - Iterate until convergence
   - Validate against multi-way tables where available

4. **Validate outputs:**
   - Check co-occurrence rates match known data
   - Verify age distribution matches SHS pattern
   - Confirm Indigenous overrepresentation maintained
   - Test that high-risk profiles emerge naturally

---

## Priority Data Gaps to Address

If refining the model, prioritize getting:

1. **Age-specific mental health rates** for all age groups (access ABS data cubes)
2. **Age-specific DV rates** (full Personal Safety Survey tables)
3. **Education and employment** by demographics (ABS Labour Force)
4. **Income source** by demographics (ABS/Centrelink data)
5. **Three-way co-occurrences** (Mental health × Drug use × DV specifics)
6. **Regional variation** in risk factors (state/territory breakdowns)
7. **Housing situation** prior to SHS entry (detailed categories)

---

## Files Provided

1. **Australian_Population_Statistics_2024-2025.md** (39KB)
   - Comprehensive report with all statistics
   - Detailed breakdowns by age, gender, Indigenous status
   - Co-occurrence rates and correlations
   - Full references and data sources
   - Data gaps and assumptions documented

2. **ipf_reference_rates.csv** (3.8KB)
   - Machine-readable rates for IPF implementation
   - Marginal distributions for all variables
   - Conditional probabilities
   - Overrepresentation factors
   - Population totals

3. **QUICK_REFERENCE_STATISTICS.md** (6.6KB)
   - Quick lookup tables
   - Key rates and comparisons
   - Model variable mappings
   - IPF constraint recommendations

4. **KEY_FINDINGS_SUMMARY.md** (this file)
   - Executive summary of top findings
   - Critical interactions for modeling
   - IPF implementation guidance
   - Data quality assessment

---

## Citation

Data compiled from:
- Australian Bureau of Statistics (ABS): National Study of Mental Health and Wellbeing 2020-2022, Population estimates, Personal Safety Survey 2021-22
- Australian Institute of Health and Welfare (AIHW): Specialist Homelessness Services Annual Report 2023-24, National Drug Strategy Household Survey 2022-2023, Family Domestic and Sexual Violence reports

Compiled: October 7, 2025 for Visionary Model - Homelessness Risk Prediction

---

## Quick Start for IPF Implementation

```python
# Load reference rates
import pandas as pd
rates = pd.read_csv('model/data/ipf_reference_rates.csv')

# Key marginals for at-risk population
marginals = {
    'indigenous': 0.29,
    'age_15_44': 0.55,
    'female': 0.60,
    'mental_health': 0.32,
    'drug_issue': 0.086,
}

# Key interactions (don't assume independence!)
interactions = {
    'mental_health_given_drug': 0.42,
    'indigenous_under_25': 0.48,
    'young_female_mental_health': 0.455,
}

# Use these to seed IPF algorithm and validate outputs
```

See full documentation for detailed implementation guidance.
