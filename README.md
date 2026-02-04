# Mental Health-Adjusted Actuarial Risk Model

A graduate-level actuarial analysis quantifying how mental health conditions affect mortality risk and life insurance pricing for young adults (ages 18-35).

## 🎯 Project Overview

This project implements both traditional statistical methods (Cox Proportional Hazards) and modern machine learning (Random Survival Forests) to:
- Quantify mortality risk associated with depression, anxiety, and substance use
- Calculate actuarially fair insurance premiums
- Analyze policy scenarios for underwriting decisions
- Provide evidence-based insights for insurance regulation

## 📊 Key Findings

- **Mental Health Impact**: Severe depression increases mortality risk by 85% (HR = 1.85)
- **Substance Use**: 90% increased mortality risk (HR = 1.90)
- **Life Expectancy**: High-risk profiles lose 2-5 years
- **Insurance Pricing**: Premiums increase 40-230% based on risk level
- **Model Performance**: Random Survival Forest achieves 90%+ C-index

## 🛠️ Technologies Used

- **Python 3.11+**
- **Statistical Analysis**: 
  - Cox Proportional Hazards (lifelines)
  - Kaplan-Meier survival curves
- **Machine Learning**:
  - Random Survival Forests (scikit-survival)
  - Permutation importance analysis
- **Data Sources**:
  - CDC 2021 Life Tables
  - SAMHSA 2022 National Survey
  - Published meta-analyses (293+ studies)

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/Mental-Health-Actuarial-Model.git
cd Mental-Health-Actuarial-Model

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Run the enhanced model
python Mental-Health-Actuarial-Model-Enhanced.py
```

**Runtime**: 3-5 minutes  
**Memory**: ~500MB RAM

## 📈 Outputs

The model generates:

### Data Files:
1. `optimal_parameters_from_real_data.json` - Extracted CDC/SAMHSA parameters
2. `calibrated_simulation_dataset.csv` - 10,000-person simulated cohort
3. `actuarial_results_summary.json` - Complete results
4. `FINAL_ACTUARIAL_REPORT.txt` - Comprehensive analysis

### Visualizations:
1. `1_cdc_mortality_rates.png` - Baseline mortality by age/sex
2. `2_mental_health_prevalence.png` - SAMHSA prevalence data
3. `3_literature_hazard_ratios.png` - Meta-analysis results
4. `4_cox_hazard_ratios.png` - Cox model results
5. `4b_random_survival_forest_analysis.png` - ML model comparison
6. `5_actuarial_analysis_comprehensive.png` - Life expectancy & premiums
7. `6_kaplan_meier_subgroups.png` - Multi-curve survival analysis

## 🎓 Methodology

### Statistical Approach:
1. **Data Extraction**: Real parameters from CDC, SAMHSA, peer-reviewed literature
2. **Simulation**: Generate 10,000-person cohort with realistic correlations
3. **Cox Regression**: Traditional survival analysis with proportional hazards
4. **Random Survival Forest**: ML ensemble for pattern detection
5. **Actuarial Calculations**: Life expectancy, pure premiums, scenario analysis

### Model Validation:
- C-index comparison (Cox vs RSF)
- Variable importance analysis
- Proportional hazards testing
- Concordance metrics

## 📊 Results Summary

| Model | C-Index | Interpretation |
|-------|---------|---------------|
| Cox Proportional Hazards | 0.82 | Good discrimination |
| Random Survival Forest | 0.91 | Excellent discrimination |
| **Improvement** | **+11%** | Significant |

### Top Risk Factors (ML Variable Importance):
1. Severe Depression (29%)
2. Substance Use (21%)
3. Anxiety (17%)
4. Moderate Depression (13%)

## 🔬 Academic Applications

- **Insurance Regulation**: Evidence for mental health parity in underwriting
- **Public Health**: Quantifies mental health mortality burden
- **Actuarial Science**: Demonstrates ML applications in survival analysis
- **Health Economics**: Premium impact assessment

## 📚 Data Sources

1. **CDC National Vital Statistics System** (2021)
   - Life tables by age, sex
   - Baseline mortality rates

2. **SAMHSA National Survey** (2022)
   - Mental health prevalence (ages 18-25)
   - Depression, anxiety, substance use rates

3. **Meta-Analyses**:
   - Walker et al. 2015 (Depression, n=293 studies)
   - Meier et al. 2016 (Anxiety, n=12 studies)
   - Sordo et al. 2017 (Opioids, n=19 studies)

## 🤝 Contributing

This is an academic project. Suggestions and improvements welcome!

## 📄 License

MIT License

## 👤 Author

**Caden Arai**  
Actuarial Science | Data Science | Machine Learning

## 🙏 Acknowledgments

- CDC National Center for Health Statistics
- SAMHSA for mental health data
- scikit-survival developers
- lifelines survival analysis library

---

⭐ **Star this repository if you find it useful!**
