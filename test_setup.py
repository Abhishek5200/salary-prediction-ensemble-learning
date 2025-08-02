#!/usr/bin/env python3
"""
Test script to verify salary prediction environment setup
Run this after following setup_instructions.md
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn imported successfully")
    except ImportError as e:
        print(f"❌ seaborn import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic machine learning functionality"""
    print("\nTesting basic ML functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Test prediction
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        print(f"✅ Model training successful, R² = {r2:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ ML functionality test failed: {e}")
        return False

def test_data_directories():
    """Check if required directories exist"""
    print("\nChecking directory structure...")
    
    import os
    
    required_dirs = ['data', 'models', 'outputs']
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            all_exist = False
    
    return all_exist

def create_sample_data():
    """Create sample data files for testing"""
    print("\nCreating sample data files...")
    
    try:
        import pandas as pd
        import numpy as np
        import os
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        train_features = pd.DataFrame({
            'jobId': [f'JOB_{i}' for i in range(n_samples)],
            'companyId': [f'COMP_{np.random.randint(1, 50)}' for _ in range(n_samples)],
            'jobType': np.random.choice(['CEO', 'CTO', 'CFO', 'MANAGER', 'SENIOR', 'JUNIOR'], n_samples),
            'degree': np.random.choice(['DOCTORAL', 'MASTERS', 'BACHELORS', 'HIGH_SCHOOL'], n_samples),
            'major': np.random.choice(['ENGINEERING', 'BUSINESS', 'COMPSCI', 'MATH'], n_samples),
            'industry': np.random.choice(['OIL', 'FINANCE', 'WEB', 'HEALTH'], n_samples),
            'yearsExperience': np.random.randint(0, 25, n_samples),
            'milesFromMetropolis': np.random.randint(0, 100, n_samples)
        })
        
        # Generate realistic salaries
        base_salary = 50
        experience_bonus = train_features['yearsExperience'] * 2
        job_bonuses = {'CEO': 50, 'CTO': 40, 'CFO': 35, 'MANAGER': 15, 'SENIOR': 10, 'JUNIOR': 0}
        job_bonus = train_features['jobType'].map(job_bonuses)
        
        salaries = base_salary + experience_bonus + job_bonus + np.random.normal(0, 5, n_samples)
        salaries = np.maximum(salaries, 10)
        
        train_salaries = pd.DataFrame({
            'jobId': train_features['jobId'],
            'salary': salaries.astype(int)
        })
        
        # Save files
        train_features.to_csv('data/train_features.csv', index=False)
        train_salaries.to_csv('data/train_salaries.csv', index=False)
        train_features.sample(100).to_csv('data/test_features.csv', index=False)
        
        print("✅ Sample data files created:")
        print(f"   - data/train_features.csv ({len(train_features)} records)")
        print(f"   - data/train_salaries.csv ({len(train_salaries)} records)")
        print(f"   - data/test_features.csv (100 records)")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def test_notebook_compatibility():
    """Test if the notebook can load and process data"""
    print("\nTesting notebook compatibility...")
    
    try:
        import pandas as pd
        
        # Try to load the sample data
        train_feat_df = pd.read_csv('data/train_features.csv')
        train_target_df = pd.read_csv('data/train_salaries.csv')
        
        # Merge data (like in the notebook)
        train_df = pd.merge(train_feat_df, train_target_df, on='jobId')
        
        print(f"✅ Data loading successful: {train_df.shape[0]} records, {train_df.shape[1]} features")
        
        # Test data preprocessing
        clean_train_df = train_df[train_df.salary > 0]
        clean_train_df = clean_train_df.drop(['jobId', 'companyId'], axis=1)
        clean_train_df = pd.get_dummies(clean_train_df)
        
        print(f"✅ Data preprocessing successful: {clean_train_df.shape[1]} features after encoding")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 SALARY PREDICTION SETUP TEST")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Package imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Basic ML functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test 3: Directory structure
    if test_data_directories():
        tests_passed += 1
    
    # Test 4: Create sample data
    if create_sample_data():
        tests_passed += 1
    
    # Test 5: Notebook compatibility
    if test_notebook_compatibility():
        tests_passed += 1
    
    # Final report
    print("\n" + "=" * 50)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your environment is ready to run the salary prediction notebook")
        print("\nNext steps:")
        print("1. Start Jupyter: jupyter notebook")
        print("2. Open Salary_PredictionNotebook.ipynb")
        print("3. Run all cells to see the analysis")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before proceeding.")
        print("📚 Refer to setup_instructions.md for detailed setup steps")

if __name__ == "__main__":
    main()