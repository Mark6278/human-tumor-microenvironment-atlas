from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_spatial_classifier(adata, target_col="leiden"):
    """Evaluates a CPU Random Forest classifier predicting spatial tumor niches."""
    X = adata.obsm["X_pca"]
    y = adata.obs[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[ML Benchmark] Spatial Niche Classification Accuracy (80/20 split): {acc:.2%}")
    return acc


def evaluate_cross_donor_generalization(adata, donor_col="donor", label_col="celltypist_label"):
    train_mask = adata.obs[donor_col] == "Donor_A"
    test_mask = adata.obs[donor_col] == "Donor_B"

    X_train = adata.obsm["X_pca_harmony"][train_mask]
    y_train = adata.obs.loc[train_mask, label_col]
    X_test = adata.obsm["X_pca_harmony"][test_mask]
    y_test = adata.obs.loc[test_mask, label_col]

    # LogisticRegression generalizes smoothly across batch-corrected PCA coordinates
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[ML Benchmark] Cross-Donor Generalization Accuracy (Train Donor_A -> Test Donor_B): {acc:.2%}")
    return acc
