import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report)

#  Config 
CSV_FILE  = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
PLOTS_DIR = "plots"
SEED      = 42

# olour palette
TEAL    = "#4A9B8F"   # primary — muted teal
LTEAL   = "#A8D5CF"   # light teal for low-importance bars
SAGE    = "#6B8F71"   # secondary green
ROSE    = "#C0756A"   # accent rose/coral
SLATE   = "#5A6475"   # dark slate for text elements
def banner(text):
    print("\n" + "─" * 55)
    print(f"  {text}")
    print("─" * 55)


# Load Data
banner("Loading Dataset")
if not os.path.exists(CSV_FILE):
    print(f"  ERROR: '{CSV_FILE}' not found.")
    print("  Place the CSV in the same folder as this script.")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"  Missing : {df.isnull().sum().sum()}")


# Preprocessing
banner("Preprocessing")
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"].astype(int)
feature_names = list(X.columns)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=SEED, stratify=y)
print(f"  Train  : {len(X_train):,} samples")
print(f"  Test   : {len(X_test):,} samples")
print(f"  Class 0 (Non-Diabetic) : {(y==0).sum():,}")
print(f"  Class 1 (Diabetic)     : {(y==1).sum():,}")

# Train Model
banner("Training Random Forest Classifier")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=SEED,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("  Training complete.")

#  Evaluate
banner("Evaluation Results")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy"  : round(accuracy_score(y_test, y_pred),  4),
    "precision" : round(precision_score(y_test, y_pred), 4),
    "recall"    : round(recall_score(y_test, y_pred),    4),
    "f1"        : round(f1_score(y_test, y_pred),        4),
    "roc_auc"   : round(roc_auc_score(y_test, y_prob),   4),
}
print(f"  Accuracy  : {metrics['accuracy']}  ({metrics['accuracy']*100:.2f}%)")
print(f"  Precision : {metrics['precision']}")
print(f"  Recall    : {metrics['recall']}")
print(f"  F1-Score  : {metrics['f1']}")
print(f"  ROC-AUC   : {metrics['roc_auc']}")
print()
print(classification_report(y_test, y_pred,
      target_names=["Non-Diabetic", "Diabetic"]))

#  Save 
banner(" Saving Model Files")
joblib.dump(model,         "model.pkl")
joblib.dump(scaler,        "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(metrics,       "metrics.pkl")
print("  model.pkl          saved")
print("  scaler.pkl         saved")
print("  feature_names.pkl  saved")
print("  metrics.pkl        saved")

# Generate Plots
banner("STEP 6 — Generating Plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.facecolor"  : "#FAFAFA",
    "axes.facecolor"    : "#FAFAFA",
})

#Plots
#  1.Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#FAFAFA")
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=["Non-Diabetic", "Diabetic"],
            yticklabels=["Non-Diabetic", "Diabetic"],
            linewidths=0.5, ax=ax, annot_kws={"size": 14})
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold",
             pad=12, color=SLATE)
ax.set_ylabel("Actual Label",    fontsize=11, color=SLATE)
ax.set_xlabel("Predicted Label", fontsize=11, color=SLATE)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {PLOTS_DIR}/confusion_matrix.png saved")

# 2.ROC-AUC Curve 
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#FAFAFA")
ax.plot(fpr, tpr, color=TEAL, lw=2.5,
        label=f"Random Forest  (AUC = {metrics['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color=ROSE, linestyle="--", lw=1.2,
        alpha=0.6, label="Random Guess")
ax.fill_between(fpr, tpr, alpha=0.10, color=TEAL)
ax.set_xlabel("False Positive Rate", fontsize=11, color=SLATE)
ax.set_ylabel("True Positive Rate",  fontsize=11, color=SLATE)
ax.set_title("ROC-AUC Curve", fontsize=14, fontweight="bold", color=SLATE)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.25, color="#CCCCCC")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {PLOTS_DIR}/roc_curve.png saved")

#3. Feature Importance 
feat_df = pd.DataFrame({
    "Feature"    : feature_names,
    "Importance" : model.feature_importances_
}).sort_values("Importance")
med    = feat_df["Importance"].median()
colors = [TEAL if v > med else LTEAL for v in feat_df["Importance"]]
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#FAFAFA")
bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
               color=colors, edgecolor="white", linewidth=0.6)
for bar, val in zip(bars, feat_df["Importance"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8, color=SLATE)
ax.axvline(med, color=ROSE, linestyle="--", lw=1.2,
           alpha=0.7, label=f"Median ({med:.3f})")
ax.set_xlabel("Importance Score", fontsize=11, color=SLATE)
ax.set_title("Feature Importance — Random Forest",
             fontsize=14, fontweight="bold", color=SLATE)
ax.legend(fontsize=9)
ax.grid(True, axis="x", alpha=0.25, color="#CCCCCC")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {PLOTS_DIR}/feature_importance.png saved")

# 4.Correlation Heatmap 
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor("#FAFAFA")
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, annot_kws={"size": 7},
            square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap", fontsize=14,
             fontweight="bold", pad=12, color=SLATE)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {PLOTS_DIR}/correlation_heatmap.png saved")

# 5. Class Distribution 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor("#FAFAFA")
labels = ["Non-Diabetic", "Diabetic"]
counts = [(y == 0).sum(), (y == 1).sum()]
axes[0].pie(counts, labels=labels, colors=[TEAL, ROSE],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Class Distribution", fontsize=13,
                  fontweight="bold", color=SLATE)
axes[0].set_facecolor("#FAFAFA")

axes[1].bar(labels, counts, color=[TEAL, ROSE],
            width=0.5, edgecolor="white", linewidth=2)
for i, v in enumerate(counts):
    axes[1].text(i, v + 200, f"{v:,}", ha="center",
                 fontsize=11, fontweight="bold", color=SLATE)
axes[1].set_title("Sample Count per Class", fontsize=13,
                  fontweight="bold", color=SLATE)
axes[1].set_ylabel("Count", fontsize=11, color=SLATE)
axes[1].set_facecolor("#F1CBCB")
axes[1].grid(True, axis="y", alpha=0.25, color="#FADDDD")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  {PLOTS_DIR}/class_distribution.png saved")
