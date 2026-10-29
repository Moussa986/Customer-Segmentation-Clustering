"""Customer Segmentation Using K-Means, Hierarchical Clustering, and PCA.

Run from the project root:
    python main.py

The Mall Customers dataset is downloaded from a public GitHub raw URL if it is
not already present in data/Mall_Customers.csv. The program saves charts,
metrics, and customer-level cluster assignments under outputs/.
"""
from pathlib import Path
import urllib.request
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_FILE = DATA_DIR / "Mall_Customers.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/hannazan/"
    "mall-customer-segmentation/main/data/Mall_Customers.csv"
)
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_dataset() -> pd.DataFrame:
    """Load the CSV, downloading it once when it is not present locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        print("Dataset not found locally. Downloading the public Mall Customers CSV...")
        try:
            request = urllib.request.Request(
                DATA_URL, headers={"User-Agent": "CustomerSegmentationStudentProject/1.0"}
            )
            with urllib.request.urlopen(request, timeout=25) as response:
                DATA_FILE.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(
                "Could not download the dataset. Check your internet connection and "
                "download Mall_Customers.csv manually from the dataset link in README.md, "
                f"then save it as {DATA_FILE}."
            ) from exc

    df = pd.read_csv(DATA_FILE)
    # Dataset copies sometimes use "Genre" instead of "Gender".
    if "Gender" not in df.columns and "Genre" in df.columns:
        df = df.rename(columns={"Genre": "Gender"})

    required = {"Age", "Annual Income (k$)", "Spending Score (1-100)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            "Use the standard Mall Customers dataset."
        )
    if df.empty:
        raise ValueError("The dataset is empty.")
    return df


def evaluate_clusters(X: np.ndarray, labels: np.ndarray) -> dict:
    """Return internal clustering metrics; invalid one-cluster cases are avoided."""
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {
            "Silhouette Score": np.nan,
            "Davies-Bouldin Index": np.nan,
            "Calinski-Harabasz Score": np.nan,
        }
    return {
        "Silhouette Score": float(silhouette_score(X, labels)),
        "Davies-Bouldin Index": float(davies_bouldin_score(X, labels)),
        "Calinski-Harabasz Score": float(calinski_harabasz_score(X, labels)),
    }


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    print("\n=== Dataset overview ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values by column:")
    print(df.isna().sum().to_string())

    # CustomerID is an identifier, not a behavioural measurement.
    clean = df.drop_duplicates().copy()
    for column in FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=FEATURES).reset_index(drop=True)
    if len(clean) < 4:
        raise ValueError("Not enough complete customer records for clustering.")

    X_raw = clean[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # EDA: distributions of the selected numerical features.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, feature in zip(axes, FEATURES):
        ax.hist(clean[feature], bins=15, edgecolor="black")
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Number of customers")
    save_plot(OUTPUT_DIR / "feature_distributions.png")

    # Elbow and silhouette analysis for k = 2..10.
    max_k = min(10, len(clean) - 1)
    candidate_ks = list(range(2, max_k + 1))
    inertias, silhouettes = [], []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(float(model.inertia_))
        if k >= 2:
            silhouettes.append(float(silhouette_score(X_scaled, model.labels_)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, max_k + 1), inertias, marker="o")
    ax.set_title("Elbow Method for K-Means")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    save_plot(OUTPUT_DIR / "elbow_method.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(candidate_ks, silhouettes, marker="o")
    ax.set_title("Silhouette Score by Number of Clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (higher is generally better)")
    save_plot(OUTPUT_DIR / "silhouette_by_k.png")

    # Select k by the highest silhouette score; record this rule in the report.
    best_k = candidate_ks[int(np.argmax(silhouettes))]
    print(f"\nSelected k for the comparison: {best_k} (highest silhouette among tested k values).")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    # PCA is used for visualisation, not as the input to the primary models.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = float(pca.explained_variance_ratio_.sum())

    for labels, title, filename in [
        (kmeans_labels, "K-Means clusters (PCA view)", "kmeans_pca_clusters.png"),
        (hierarchical_labels, "Hierarchical clusters (PCA view)", "hierarchical_pca_clusters.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=38, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
        ax.text(
            0.02, 0.02, f"Explained variance by first 2 PCs: {explained:.1%}",
            transform=ax.transAxes, fontsize=8
        )
        save_plot(OUTPUT_DIR / filename)

    # Dendrogram from a sample to keep labels readable.
    sample_n = min(60, len(X_scaled))
    Z = linkage(X_scaled[:sample_n], method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, no_labels=True, ax=ax, color_threshold=None)
    ax.set_title(f"Hierarchical Clustering Dendrogram (first {sample_n} records)")
    ax.set_xlabel("Sampled customer observations")
    ax.set_ylabel("Ward linkage distance")
    save_plot(OUTPUT_DIR / "hierarchical_dendrogram.png")

    metrics = []
    for algorithm, labels in [
        ("K-Means", kmeans_labels),
        ("Agglomerative Hierarchical", hierarchical_labels),
    ]:
        row = {"Algorithm": algorithm, "Number of clusters": len(np.unique(labels))}
        row.update(evaluate_clusters(X_scaled, labels))
        metrics.append(row)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "clustering_metrics.csv", index=False)

    results = clean.copy()
    results["KMeans_Cluster"] = kmeans_labels
    results["Hierarchical_Cluster"] = hierarchical_labels
    results.to_csv(OUTPUT_DIR / "customer_cluster_assignments.csv", index=False)

    # Descriptive cluster profiles support interpretation without inventing names.
    profile_rows = []
    for algorithm, label_col in [
        ("K-Means", "KMeans_Cluster"),
        ("Agglomerative Hierarchical", "Hierarchical_Cluster"),
    ]:
        grouped = results.groupby(label_col)[FEATURES].agg(["count", "mean"])
        for cluster_id, row in grouped.iterrows():
            profile_rows.append({
                "Algorithm": algorithm,
                "Cluster": int(cluster_id),
                "Customer count": int(row[(FEATURES[0], "count")]),
                "Mean age": float(row[(FEATURES[0], "mean")]),
                "Mean annual income (k$)": float(row[(FEATURES[1], "mean")]),
                "Mean spending score": float(row[(FEATURES[2], "mean")]),
            })
    pd.DataFrame(profile_rows).to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)

    print("\n=== Internal evaluation metrics ===")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nPCA variance explained by two components: {explained:.2%}")
    print("\nCluster profiles saved to outputs/cluster_profiles.csv")
    print("Customer-level labels saved to outputs/customer_cluster_assignments.csv")
    print(f"Charts and result files saved in: {OUTPUT_DIR}")
    print("\nInterpretation note: cluster numbers are labels, not rankings or ground-truth categories.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
