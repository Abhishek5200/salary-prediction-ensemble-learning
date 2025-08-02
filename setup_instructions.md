# Local Setup Instructions for Salary Prediction Notebook

## Prerequisites
- Python 3.7+ installed on your system
- Git (optional, for cloning repositories)

## Step 1: Create Project Directory
```bash
mkdir salary_prediction_project
cd salary_prediction_project
```

## Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Required Packages
```bash
pip install pandas numpy scikit-learn seaborn matplotlib scipy jupyter
```

## Step 4: Create Directory Structure
```bash
mkdir data
mkdir models
mkdir outputs
```

Your folder structure should look like:
```
salary_prediction_project/
├── venv/
├── data/
│   ├── train_features.csv
│   ├── train_salaries.csv
│   └── test_features.csv
├── models/
├── outputs/
├── Salary_PredictionNotebook.ipynb
└── requirements.txt
```

## Step 5: Create Requirements File
Create `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
seaborn>=0.11.0
matplotlib>=3.4.0
scipy>=1.7.0
jupyter>=1.0.0
```

## Step 6: Download Dataset
You need to obtain the original dataset files:
- `train_features.csv` - Training features (1M records)
- `train_salaries.csv` - Training target values
- `test_features.csv` - Test features for predictions

**Possible sources:**
1. Original GitHub repository where you found this notebook
2. Kaggle datasets (search for "salary prediction")
3. Contact the notebook author: abhisheksingh987666@gmail.com

## Step 7: Fix Python 2 to 3 Compatibility Issues

Replace these lines in the notebook:

**Old (Python 2):**
```python
print 'Text'
from sklearn.externals import joblib
```

**New (Python 3):**
```python
print('Text')
import joblib
```

## Step 8: Run the Notebook
```bash
# Start Jupyter
jupyter notebook

# Open Salary_PredictionNotebook.ipynb in the browser
```

## Alternative: Use Sample Data for Testing

If you can't get the original dataset, you can create sample data for testing:

```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 10000

# Generate features
train_features = pd.DataFrame({
    'jobId': [f'JOB_{i}' for i in range(n_samples)],
    'companyId': [f'COMP_{np.random.randint(1, 100)}' for _ in range(n_samples)],
    'jobType': np.random.choice(['CEO', 'CTO', 'CFO', 'VICE_PRESIDENT', 'MANAGER', 'SENIOR', 'JUNIOR', 'JANITOR'], n_samples),
    'degree': np.random.choice(['DOCTORAL', 'MASTERS', 'BACHELORS', 'HIGH_SCHOOL', 'NONE'], n_samples),
    'major': np.random.choice(['ENGINEERING', 'BUSINESS', 'COMPSCI', 'MATH', 'PHYSICS', 'CHEMISTRY', 'BIOLOGY', 'LITERATURE', 'NONE'], n_samples),
    'industry': np.random.choice(['OIL', 'FINANCE', 'WEB', 'AUTO', 'HEALTH', 'EDUCATION', 'SERVICE'], n_samples),
    'yearsExperience': np.random.randint(0, 25, n_samples),
    'milesFromMetropolis': np.random.randint(0, 100, n_samples)
})

# Generate realistic salaries
base_salary = 50
experience_bonus = train_features['yearsExperience'] * 2
job_bonuses = {'CEO': 50, 'CTO': 40, 'CFO': 35, 'VICE_PRESIDENT': 25, 'MANAGER': 15, 'SENIOR': 10, 'JUNIOR': 0, 'JANITOR': -10}
job_bonus = train_features['jobType'].map(job_bonuses)

# Add noise and ensure positive values
salaries = base_salary + experience_bonus + job_bonus + np.random.normal(0, 5, n_samples)
salaries = np.maximum(salaries, 10)

train_salaries = pd.DataFrame({
    'jobId': train_features['jobId'],
    'salary': salaries.astype(int)
})

# Save sample data
train_features.to_csv('data/train_features.csv', index=False)
train_salaries.to_csv('data/train_salaries.csv', index=False)
train_features.sample(1000).to_csv('data/test_features.csv', index=False)

print("Sample data created successfully!")
```

## Troubleshooting

### Common Issues:

1. **Module not found errors:**
   ```bash
   pip install --upgrade pip
   pip install [missing_package]
   ```

2. **Jupyter not starting:**
   ```bash
   pip install --upgrade jupyter
   jupyter --version
   ```

3. **Matplotlib display issues:**
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'
   ```

4. **Memory issues with large dataset:**
   - Use smaller sample size for testing
   - Process data in chunks
   - Use `pd.read_csv(chunksize=1000)`

### Performance Tips:
- Use `%matplotlib inline` in Jupyter for plots
- Set `pd.options.display.max_columns = None` for full dataframe display
- Use `warnings.filterwarnings('ignore')` to suppress warnings

## Running Successfully
Once set up correctly, you should see:
- Data loading without errors
- Visualizations displaying properly
- Model training completing
- Performance metrics showing ~76% accuracy
- Model saving successfully

## Next Steps After Setup
1. Explore the data with different visualizations
2. Try different model parameters
3. Add new features or data sources
4. Deploy the model for real predictions
5. Create a web interface for the model