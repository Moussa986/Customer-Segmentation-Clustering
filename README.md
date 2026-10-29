# Customer Segmentation Using Clustering

A B.Sc. Computer Science Data Mining mini-project that explores customer groups using **K-Means**, **Agglomerative Hierarchical Clustering**, and **Principal Component Analysis (PCA)**.

## Project objective

Use customer age, annual income, and spending score to discover groups with similar profiles. This is an **unsupervised learning** task: there is no predefined target label.

## Dataset

This project uses the public **Mall Customers** dataset (about 200 customer records). It commonly contains `CustomerID`, `Gender` (some copies use `Genre`), `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.

- Dataset page: https://github.com/hannazan/mall-customer-segmentation/blob/main/data/Mall_Customers.csv
- Direct CSV source used by the script: https://raw.githubusercontent.com/hannazan/mall-customer-segmentation/main/data/Mall_Customers.csv

The script downloads the CSV to `data/Mall_Customers.csv` on its first run if it is not present. An internet connection is required for that first download. You can also download the CSV manually from the source page and save it at that path.

`CustomerID` is not used as a model feature because it is an identifier. The primary clustering features are age, annual income, and spending score. These numeric features are standardized before distance-based clustering. Gender is retained in the original data but is not included in the primary model.

## Algorithms and analysis

1. **Data inspection and preparation** — inspect rows, columns, missing values, remove duplicate records, coerce selected feature columns to numeric, and remove records missing required feature values.
2. **K-Means** — fit models across a range of `k` values, plot inertia using the Elbow Method, and plot silhouette scores. For the algorithm comparison, the script chooses the tested `k` with the highest silhouette score.
3. **Agglomerative Hierarchical Clustering** — use Ward linkage with the same selected cluster count and create a dendrogram for a sample of records.
4. **PCA** — project standardized three-feature data to two components for visualisation. PCA is not used to fit the primary clustering models.
5. **Evaluation** — compare Silhouette Score (higher is generally better), Davies–Bouldin Index (lower is generally better), and Calinski–Harabasz Score (higher is generally better). These are internal metrics, not proof that segments are objectively correct.
6. **Cluster profiling** — calculate each cluster's customer count and mean age, income, and spending score.

## Folder structure

```text
Customer_Segmentation_Clustering_Project/
├── data/
│   └── .gitkeep
├── outputs/                 # created when the program runs; ignored by Git
├── docs/
│   ├── PROJECT_REPORT.md
│   └── VIVA_QUESTIONS.md
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run on Windows (VS Code / PowerShell)

Open this folder in VS Code. In the terminal, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks environment activation, run this in the same terminal session, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Output files

After a successful run, the `outputs/` folder contains:

- `feature_distributions.png`
- `elbow_method.png`
- `silhouette_by_k.png`
- `kmeans_pca_clusters.png`
- `hierarchical_pca_clusters.png`
- `hierarchical_dendrogram.png`
- `clustering_metrics.csv`
- `customer_cluster_assignments.csv`
- `cluster_profiles.csv`

The console prints the dataset overview, selected `k`, internal evaluation metrics, and PCA variance explained. **Do not report numerical findings until you have run the project and checked the generated output files.** Cluster IDs are arbitrary labels, not rankings.

## Limitations

- The dataset is small and commonly used for teaching; it may not represent all mall customers.
- Spending score is provided by the dataset and its real-world calculation is not documented here.
- Internal metrics compare cluster structure but do not establish business value or causal effects.
- PCA plots compress three features into two dimensions, so some information is lost.

## Academic integrity

Run the code yourself, review the actual generated metrics and plots, and write your final interpretation from those outputs. The report in `docs/PROJECT_REPORT.md` is a structured starting point and deliberately does not invent measured results.
