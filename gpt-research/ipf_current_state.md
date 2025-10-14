# Current IPF Implementation State

## Visual Representation of Multi-Dimensional Data Space

### Current Implementation: 2D IPF Only

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENDER × AGE GROUP                           │
│                   ✅ IPF CONVERGED                               │
├─────────────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬─┤
│             │ 0-9  │10-17 │18-24 │25-34 │35-44 │45-54 │55-64 │65+│
├─────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼─┤
│ Male        │ 21.7%│ 14.4%│ 12.1%│ 11.6%│ 14.1%│ 13.1%│ 8.1% │4.9│
│ Female      │ 13.1%│ 11.1%│ 16.2%│ 20.2%│ 19.6%│ 12.0%│ 3.6% │3.6│
└─────────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴─┘
```

### All Other Features: INDEPENDENT RANDOM ASSIGNMENT ❌

```
Feature Dimensions (NOT converged via IPF):

┌────────────────────────────────────┐
│ Indigenous Status                  │
│ ❌ Random: 29.7% probability       │
│ ❌ No cross-constraints            │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Mental Health Issue                │
│ ❌ Random by Gender:               │
│    Male: 26.1%                     │
│    Female: 29.6%                   │
│ ❌ No age/indigenous constraints   │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Drug/Alcohol Issue                 │
│ ❌ Random by Gender:               │
│    Male: 9.7%                      │
│    Female: 5.8%                    │
│ ❌ No co-occurrence with MH        │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Geographic Location                │
│ ❌ Random: 15 categories           │
│ ❌ No gender/age/homeless links    │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Income Source (15+)                │
│ ❌ Random: 77% govt, 12% employ... │
│ ❌ No employment/education links   │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Employment Status (15+)            │
│ ❌ Random: 51% unemployed...       │
│ ❌ No income/education links       │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Education Level                    │
│ ❌ Random distribution             │
│ ❌ No age/employment links         │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Domestic Violence                  │
│ ❌❌ NOT INCLUDED AT ALL! ❌❌      │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ Homelessness Status (TARGET)       │
│ ❌ Risk score + threshold          │
│ ✓ Correct 53.05% ratio            │
│ ❌ No IPF convergence              │
└────────────────────────────────────┘
```

## Multi-Dimensional Space Visualization

### What We Have Now (Simplified 3D View):

```
                    AGE
                     ↑
                     │
        ┌────────────┼────────────┐
       /│            │            │\
      / │   ✅ IPF   │            │ \
     /  │  CONVERGED │            │  \
    /   │            │            │   \
   /    └────────────┼────────────┘    \
  /                  │                  \
 ←──────────────── GENDER ──────────────→
  \                                    /
   \     All other features:         /
    \    - Indigenous               /
     \   - Mental Health           /
      \  - Drug Use               /
       \ - Location              /
        \- Income               /
         - Employment          /
          - Education         /
           - DV (missing!)   /
            - Homeless ❌   /
             └────────────┘
                   ↓
            (Random assignment,
         no convergence to margins)
```

### What We NEED (Full Multi-Dimensional IPF):

```
                                    AGE (8 groups)
                                     ↑
                                     │
                            ╔════════╪════════╗
                           ║   IPF CONVERGED  ║
                          ║    ON ALL AXES   ║
                         ║                   ║
                        ║                     ║
    INDIGENOUS ←───────╬─────────┼───────────╬────────→ GENDER
                      ║          │            ║
                     ║           │             ║
        MENTAL     ║            │              ║   DRUG USE
        HEALTH ←──╬─────────────┼──────────────╬───→
                 ║              │               ║
                ║               │                ║
    LOCATION ←─╬────────────────┼────────────────╬──→ EMPLOYMENT
              ║                 │                 ║
             ║                  │                  ║
   INCOME ←─╬───────────────────┼───────────────────╬─→ EDUCATION
           ║                    │                    ║
          ║                     │                     ║
         ║         ↓ DOMESTIC VIOLENCE ↓              ║
        ║                      │                       ║
       ║                       │                        ║
      ║                        ↓                         ║
     ║                   HOMELESSNESS                    ║
    ║                    (TARGET VAR)                     ║
   ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
           ALL MARGINS CONVERGED VIA IPF
```

## Current Data Flow:

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Generate Base Population                               │
├─────────────────────────────────────────────────────────────────┤
│  • Gender: Random (38.8% M, 61.2% F)                           │
│  • Age: Random within gender-specific age distribution         │
│  • Indigenous: Random (29.7%)                                   │
│  • Mental Health: Random by gender                             │
│  • Drug Use: Random by gender                                  │
│  • Location: Random (15 categories)                            │
│  • Income: Random (if age 15+)                                 │
│  • Employment: Random (if age 15+)                             │
│  • Education: Random (age-dependent)                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Apply IPF (Gender × Age ONLY)                          │
├─────────────────────────────────────────────────────────────────┤
│  ✅ Create 2D contingency table (Gender × Age)                 │
│  ✅ Apply IPF to match SHS marginals                           │
│  ✅ Calculate weights for resampling                           │
│  ❌ All other features remain randomly distributed             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Assign Homelessness via Risk Score                     │
├─────────────────────────────────────────────────────────────────┤
│  • Calculate risk = f(age, indigenous, MH, drugs, income...)   │
│  • Add random noise                                             │
│  • Set threshold at 53.05 percentile                           │
│  ❌ NOT using IPF - just threshold matching                    │
│  ❌ Doesn't preserve cross-tabs with other features           │
└─────────────────────────────────────────────────────────────────┘
```

## Problem Illustration:

### Example Issue #1: Mental Health × Homelessness

```
What SHS Data Might Show:
┌──────────────────┬───────────┬─────────────┐
│                  │ Homeless  │ Not Homeless│
├──────────────────┼───────────┼─────────────┤
│ Has Mental Health│   45%     │    20%      │
│ No Mental Health │   55%     │    80%      │
└──────────────────┴───────────┴─────────────┘

What Our Code Produces (random):
┌──────────────────┬───────────┬─────────────┐
│                  │ Homeless  │ Not Homeless│
├──────────────────┼───────────┼─────────────┤
│ Has Mental Health│   28.3%   │    28.3%    │  ← Same rate!
│ No Mental Health │   71.7%   │    71.7%    │  ← No correlation!
└──────────────────┴───────────┴─────────────┘
```

### Example Issue #2: Employment × Income (should be correlated!)

```
What Should Happen:
┌──────────────────┬──────────────┬─────────┬──────────┐
│                  │ Govt Payment │ Employed│ No Income│
├──────────────────┼──────────────┼─────────┼──────────┤
│ Employed FT/PT   │   Low %      │  High % │   0%     │
│ Unemployed       │   High %     │   0%    │  Some %  │
│ Not in Labour    │   High %     │   0%    │  Some %  │
└──────────────────┴──────────────┴─────────┴──────────┘

What Our Code Produces:
┌──────────────────┬──────────────┬─────────┬──────────┐
│                  │ Govt Payment │ Employed│ No Income│
├──────────────────┼──────────────┼─────────┼──────────┤
│ Employed FT/PT   │    77%       │   12%   │   9.5%   │  ← Illogical!
│ Unemployed       │    77%       │   12%   │   9.5%   │  ← Same dist!
│ Not in Labour    │    77%       │   12%   │   9.5%   │  ← No link!
└──────────────────┴──────────────┴─────────┴──────────┘
```

## Summary Table:

| Feature               | Current State          | IPF Applied? | Cross-constraints? |
|-----------------------|------------------------|--------------|-------------------|
| Gender                | ✅ Correct marginals   | ✅ Yes       | ✅ With Age       |
| Age                   | ✅ Correct marginals   | ✅ Yes       | ✅ With Gender    |
| Indigenous            | ✅ Correct marginal    | ❌ No        | ❌ Independent    |
| Mental Health         | ✅ Correct by gender   | ❌ No        | ❌ Independent    |
| Drug Use              | ✅ Correct by gender   | ❌ No        | ❌ Independent    |
| Domestic Violence     | ❌ NOT INCLUDED        | ❌ No        | ❌ N/A            |
| Location              | ✅ Correct marginal    | ❌ No        | ❌ Independent    |
| Income Source         | ✅ Correct marginal    | ❌ No        | ❌ Independent    |
| Employment            | ✅ Correct marginal    | ❌ No        | ❌ Independent    |
| Education             | ⚠️ Estimated           | ❌ No        | ❌ Independent    |
| **Homelessness**      | ✅ Correct ratio       | ❌ No        | ⚠️ Via risk score |

## Conclusion:

**Current approach**: 2D IPF (Gender × Age) + independent random assignment for everything else
**Needed approach**: Multi-dimensional IPF with all available cross-tabulations to preserve realistic correlations between features
