######################################################
#  Student Name: Truong Pham
#  Student ID: W30602308
#  Course Code: CSCI 460 -- Fall 2025
#  Assignment Due Date: 10/14/2025
#  GitHub Link: https://github.com/tpham0322/CSCI460_W30602308_Programming_Assignment1
######################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv('bank-full.csv', sep=';')

# Store mappings before encoding
mappings = {}

#categorical columns to integers
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'poutcome', 'y']

for col in categorical_cols:
    df[col] = df[col].astype('category')
    mappings[col] = dict(enumerate(df[col].cat.categories))
    df[col] = df[col].cat.codes

# Remove rows with missing values
df = df.dropna()

# Set X and y
X = df.drop('y', axis=1)
y = df['y']

def run_trials(X, y, train_size, n_trials=10):
    accs, f1s = [], []
    sss = StratifiedShuffleSplit(n_splits=n_trials, train_size=train_size, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
    return accs, f1s

# -----------------------
# Task II: 70/30 split, 10 trials
# -----------------------
accs, f1s = run_trials(X, y, train_size=0.7, n_trials=10)

print("\n=== Task II: 70/30 split (10 trials) ===")
for i, (acc, f1) in enumerate(zip(accs, f1s), start=1):
    print(f"Trial {i:2d}: Accuracy={acc:.4f}, F1={f1:.4f}")

print(f"\nAverage Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Average F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# -----------------------
# Task III: Varying train sizes
# -----------------------
train_sizes = np.arange(0.1, 1.0, 0.1)
results = []
for ts in train_sizes:
    accs, f1s = run_trials(X, y, train_size=ts, n_trials=10)
    results.append({
        'train_size': ts,
        'acc_mean': np.mean(accs),
        'acc_std': np.std(accs), 
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s)
    })

# Print table
print("\n=== Task III: Varying Train Sizes ===")
print("Train Size\tAcc Mean\tAcc Std\t\tF1 Mean\t\tF1 Std")
for r in results:
    print(f"{r['train_size']:.1f}\t\t{r['acc_mean']:.4f}\t\t{r['acc_std']:.4f}\t\t{r['f1_mean']:.4f}\t\t{r['f1_std']:.4f}")


import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.errorbar(train_sizes, [r['acc_mean'] for r in results],
             yerr=[r['acc_std'] for r in results], label='Accuracy', marker='o')
plt.errorbar(train_sizes, [r['f1_mean'] for r in results],
             yerr=[r['f1_std'] for r in results], label='F1-score', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Naive Bayes Performance vs Training Set Size')
plt.legend()
plt.grid(True)
plt.show()
