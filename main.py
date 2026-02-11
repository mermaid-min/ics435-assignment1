"""
Breast Cancer Classification using KNN, Decision Tree, and Random Forest
ICS 435 - Machine Learning Assignment 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

# Make outputs directory
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset loaded successfully")
print(f"Total samples: {len(X)}")
print(f"Features: {len(X[0])}")
print(f"Malignant: {sum(y==0)}, Benign: {sum(y==1)}")

# Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Scale features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results
all_results = {}

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{model_name}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm
    }

# ==================== BASELINE MODELS ====================
print("\n" + "-"*50)
print("BASELINE MODELS")
print("-"*50)

# KNN with k=5
print("\nTraining KNN (k=5)...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
knn_results = evaluate_model(y_test, y_pred_knn, "KNN (k=5)")
all_results['KNN_k5'] = knn_results

# Decision Tree
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_results = evaluate_model(y_test, y_pred_dt, "Decision Tree")
all_results['DT_default'] = dt_results

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_results = evaluate_model(y_test, y_pred_rf, "Random Forest (100 trees)")
all_results['RF_100trees'] = rf_results

# Plot confusion matrices for baseline models
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models = [
    ("KNN (k=5)", knn_results['confusion_matrix']),
    ("Decision Tree", dt_results['confusion_matrix']),
    ("Random Forest", rf_results['confusion_matrix'])
]

for idx, (name, cm) in enumerate(models):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                ax=axes[idx])
    axes[idx].set_title(f'{name}')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\nSaved confusion matrices")
plt.close()

# ==================== KNN ABLATION STUDY ====================
print("\n" + "-"*50)
print("KNN ABLATION STUDY - varying k")
print("-"*50)

k_values = [1, 3, 5, 7, 9, 11, 15, 20]
knn_results_list = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)
    y_pred = knn_model.predict(X_test_scaled)
    
    results = evaluate_model(y_test, y_pred, f"KNN (k={k})")
    knn_results_list.append({
        'k': k,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score']
    })
    all_results[f'KNN_k{k}'] = results

# Plot KNN results
knn_df = pd.DataFrame(knn_results_list)
plt.figure(figsize=(10, 6))
plt.plot(knn_df['k'], knn_df['accuracy'], marker='o', label='Accuracy', linewidth=2)
plt.plot(knn_df['k'], knn_df['precision'], marker='s', label='Precision', linewidth=2)
plt.plot(knn_df['k'], knn_df['recall'], marker='^', label='Recall', linewidth=2)
plt.plot(knn_df['k'], knn_df['f1_score'], marker='d', label='F1-Score', linewidth=2)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('KNN Performance vs k')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/ablation_knn.png', dpi=300, bbox_inches='tight')
print("\nSaved KNN ablation plot")
plt.close()

knn_df.to_csv('outputs/ablation_knn.csv', index=False)

# ==================== DECISION TREE ABLATION STUDY ====================
print("\n" + "-"*50)
print("DECISION TREE ABLATION STUDY - varying max_depth")
print("-"*50)

depth_values = [None, 3, 5, 7, 10, 15, 20]
dt_results_list = []

for depth in depth_values:
    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    
    depth_str = str(depth) if depth else 'None'
    results = evaluate_model(y_test, y_pred, f"Decision Tree (depth={depth_str})")
    dt_results_list.append({
        'max_depth': depth_str,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score']
    })
    all_results[f'DT_depth{depth_str}'] = results

# Plot DT results
dt_df = pd.DataFrame(dt_results_list)
plt.figure(figsize=(10, 6))
x_pos = range(len(dt_df))
plt.plot(x_pos, dt_df['accuracy'], marker='o', label='Accuracy', linewidth=2)
plt.plot(x_pos, dt_df['precision'], marker='s', label='Precision', linewidth=2)
plt.plot(x_pos, dt_df['recall'], marker='^', label='Recall', linewidth=2)
plt.plot(x_pos, dt_df['f1_score'], marker='d', label='F1-Score', linewidth=2)
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Decision Tree Performance vs Max Depth')
plt.xticks(x_pos, dt_df['max_depth'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/ablation_dt.png', dpi=300, bbox_inches='tight')
print("\nSaved Decision Tree ablation plot")
plt.close()

dt_df.to_csv('outputs/ablation_dt.csv', index=False)

# ==================== RANDOM FOREST ABLATION STUDY ====================
print("\n" + "-"*50)
print("RANDOM FOREST ABLATION STUDY - varying max_depth")
print("-"*50)

rf_depth_values = [None, 5, 10, 15, 20]
rf_results_list = []

for depth in rf_depth_values:
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    depth_str = str(depth) if depth else 'None'
    results = evaluate_model(y_test, y_pred, f"Random Forest (depth={depth_str})")
    rf_results_list.append({
        'max_depth': depth_str,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score']
    })
    all_results[f'RF_depth{depth_str}'] = results

# Plot RF results
rf_df = pd.DataFrame(rf_results_list)
plt.figure(figsize=(10, 6))
x_pos = range(len(rf_df))
plt.plot(x_pos, rf_df['accuracy'], marker='o', label='Accuracy', linewidth=2)
plt.plot(x_pos, rf_df['precision'], marker='s', label='Precision', linewidth=2)
plt.plot(x_pos, rf_df['recall'], marker='^', label='Recall', linewidth=2)
plt.plot(x_pos, rf_df['f1_score'], marker='d', label='F1-Score', linewidth=2)
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Random Forest Performance vs Max Depth')
plt.xticks(x_pos, rf_df['max_depth'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/ablation_rf.png', dpi=300, bbox_inches='tight')
print("\nSaved Random Forest ablation plot")
plt.close()

rf_df.to_csv('outputs/ablation_rf.csv', index=False)

# ==================== FINAL COMPARISON ====================
print("\n" + "-"*50)
print("FINAL COMPARISON")
print("-"*50)

# Create comparison table
comparison_data = []
for model_name, metrics in all_results.items():
    comparison_data.append({
        'Model': model_name,
        'Accuracy': f"{metrics['accuracy']:.3f}",
        'Precision': f"{metrics['precision']:.3f}",
        'Recall': f"{metrics['recall']:.3f}",
        'F1-Score': f"{metrics['f1_score']:.3f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))
comparison_df.to_csv('outputs/model_comparison.csv', index=False)

# Plot overall comparison
fig, ax = plt.subplots(figsize=(14, 6))
model_names = list(all_results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1_score']

x = np.arange(len(model_names))
width = 0.2

for i, metric in enumerate(metrics):
    values = [all_results[model][metric] for model in model_names]
    ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.9, 1.0])
plt.tight_layout()
plt.savefig('outputs/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved metrics comparison plot")
plt.close()

print("\n" + "-"*50)
print("ALL DONE! Results saved in 'outputs' folder")
print("-"*50)