# Workforce Data Analysis

## Overview
This project focuses on workforce data analysis. The goal is to clean, impute, and enrich employee datasets, perform exploratory data analysis (EDA), visualize trends, and prepare fully processed datasets for reporting and analytics insights.

## Features
- Data Cleaning & Imputation: Handle missing values for Age, DOB, and other key employee attributes.
- Feature Engineering: Create derived features like AgeGroup, TenureDays, JobFamily, SeniorityLevel, and DivisionGroup.
- Exploratory Data Analysis (EDA):
  - Age distribution by JobFamily, SeniorityLevel, and Region.
  - Tenure distribution analysis.
  - Performance Score vs Age and SeniorityLevel.
  - Categorical feature distributions (bar and pie charts).
- Correlation Analysis: Identify numeric feature correlations with Age and other variables.
- Visualization: Boxplots, histograms, KDE plots, and heatmaps for insights.
- Dataset Versions: Cleaned1, Cleaned2, and Cleaned3 versions for traceability and downstream analysis.

## Dataset
The dataset includes employee records with key fields: EmpID, FirstName, LastName, DOB, Age, StartDate, ExitDate, TenureDays, TerminationType, TerminationDescription, JobFamily, SeniorityLevel, JobFunctionGroup, DivisionGroup, Region, GenderCode, MaritalDesc, Performance Score, PayZone, and more. Sensitive information like emails is removed in the cleaned datasets.

## Project Structure
- phase1_data_loading.py
- phase2_data_cleaning.py
- phase3_date_conversion.py
- phase4_categorical_mapping.py
- phase5_missing_analysis.py
- phase6_age_dob_imputation.py
- phase7_birthyear_estimation.py
- phase8_eda_visualizations.py
- phase9_cleaned_dataset_versions.py
- phase10_final_missing_check.py
- phase11_advanced_visualizations.py
- output/
  - cleaned_data_v1.csv
  - cleaned_data_v2.csv
  - cleaned_data_v3.csv
  - phase*_*.csv
  - plots/

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/workforce-data-analysis.git
cd workforce-data-analysis
