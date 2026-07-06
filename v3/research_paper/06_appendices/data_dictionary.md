# Data Dictionary

## Source
- **File**: `data/raw/employee_data.csv`
- **Records**: 3,000 employees
- **Raw columns**: 26
- **Engineered features**: 27
- **Total features**: 53
- **Data type**: Synthetic (simulated HR data)
- **Outcomes modeled**: 5

## Raw Data Columns (26)

| Column | Type | Non-Null | Unique | Description |
|--------|------|----------|--------|-------------|
| EmpID | int64 | 3000 | 3000 | Unique employee identifier |
| FirstName | object | 3000 | 2985 | First name (PII - synthetic) |
| LastName | object | 3000 | 2992 | Last name (PII - synthetic) |
| StartDate | object | 3000 | 2519 | Employment start date |
| ExitDate | object | 1533 | 1467 | Termination/exit date |
| Title | object | 3000 | 32 | Job title |
| Supervisor | object | 3000 | 2952 | Manager/supervisor name |
| ADEmail | object | 3000 | 3000 | Email address (PII - synthetic) |
| BusinessUnit | object | 3000 | 10 | Business unit code |
| EmployeeStatus | object | 3000 | 3 | Active / Terminated / LOA |
| EmployeeType | object | 3000 | 2 | Full-time / Part-time / Contract |
| PayZone | object | 3000 | 4 | Compensation zone (A/B/C/D) |
| EmployeeClassificationType | object | 3000 | 2 | Classification type |
| TerminationType | object | 3000 | 5 | Type of termination |
| TerminationDescription | object | 1533 | 1467 | Termination reason text |
| DepartmentType | object | 3000 | 5 | Department name |
| Division | object | 3000 | 25 | Division name |
| DOB | object | 3000* | 2997 | Date of birth (60.8% unparseable) |
| State | object | 3000 | 28 | US state code |
| JobFunctionDescription | object | 3000 | 83 | Job function description |
| GenderCode | object | 3000 | 2 | Gender code |
| LocationCode | int64 | 3000 | 4 | Location ID |
| RaceDesc | object | 3000 | 5 | Race/Ethnicity |
| MaritalDesc | object | 3000 | 4 | Marital status |
| Performance Score | object | 3000 | 4 | Performance rating label |
| Current Employee Rating | int64 | 3000 | 5 | Numeric rating (1-5) |

## Engineered Features (27)

Features derived from raw data during Phase 2 of the pipeline:

### Demographic
- **Age**: Derived from DOB where parseable
- **Generation**: Categorization by birth year (Boomer, Gen X, Millennial, Gen Z)
- **CareerStage**: Numeric career stage based on age
- **CareerStageLabel**: Categorical career stage label

### Tenure & Employment
- **TenureDays**: Days from StartDate to ExitDate or today
- **TenureYears**: TenureDays / 365.25
- **TenureCategory**: Bucketed tenure ranges
- **TenureGroup**: Grouped tenure label
- **IsLongTenure**: Binary flag for tenure > 10 years
- **TenureVsAvg**: Tenure relative to department average

### Role & Hierarchy
- **SeniorityLevel**: Ordinal seniority bucket (1-5)
- **JobFamily**: Job family grouping (7 labels from titles)
- **DivisionGroup**: Grouped division
- **Region**: US region mapped from state
- **IsExecutive**: Executive role flag
- **IsManager**: Manager role flag
- **IsIC**: Individual contributor flag
- **SpanOfControl**: Number of direct reports
- **OrgLevel**: Organizational hierarchy depth
- **DistanceFromRoot**: Distance from CEO in org tree
- **PromotionReadiness**: Derived promotion likelihood
- **PromotionRate**: Historical promotion frequency
- **PromotionLag**: Time since last promotion

### Compensation & Diversity
- **PayEquityRatio**: Salary / department median salary
- **DeptGenderRatio**: Gender ratio within department
- **DeptDiversityScore**: Simpson diversity index per department
- **IntersectionalID**: Combined Gender x Race x Department identifier

### Temporal
- **StartYear**: Year of employment start
- **StartQuarter**: Quarter of employment start
- **ExitYear**: Year of exit (0 if active)
- **ExitQuarter**: Quarter of exit (0 if active)

### Performance
- **PerfScore**: Mean of Current Employee Rating (3-year proxy)
- **EngagementFlag**: Binary flag for low engagement risk
- **EarlyTenureFlag**: Binary flag for early tenure risk

## Feature Encoding

For modeling, categorical features were one-hot encoded. Ordinal features
(SeniorityLevel) were label-encoded. Numeric features were standardized
(z-score) for logistic regression.

## Outcomes (5)

| Outcome | Type | Description |
|---------|------|-------------|
| is_terminated | Binary | 1 = terminated, 0 = active |
| PerfScore | Numeric (1-5) | Mean performance rating |
| PayZone_encoded | Categorical (4) | Compensation zone (A/B/C/D) |
| is_minority_dept | Binary | 1 = employee in minority-majority dept |
| SeniorityLevel | Ordinal (1-5) | Seniority level |
