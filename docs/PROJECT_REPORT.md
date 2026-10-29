# Data Mining Mini-Project Report
## Customer Segmentation Using K-Means, Hierarchical Clustering, and PCA

> **Student note:** Fill in your university cover-page details and complete the Results section only after running `python main.py`. Do not submit placeholders as experimental results.

## 1. Introduction

Customer segmentation is the process of grouping customers who share similar characteristics. Retail organisations can use segments to explore purchasing patterns and plan differentiated communication, offers, and services. This project applies unsupervised data mining methods to a public Mall Customers dataset.

## 2. Problem Statement

A mall has customer-level demographic and behavioural measurements but no predefined customer-category label. The task is to investigate whether customers can be grouped using age, annual income, and spending score, and to compare two clustering methods.

## 3. Objectives

- Inspect and prepare the dataset for clustering.
- Apply feature standardisation before distance-based algorithms.
- Explore candidate cluster counts using the Elbow Method and silhouette scores.
- Fit K-Means and agglomerative hierarchical clustering.
- Use a dendrogram to illustrate hierarchical merging.
- Apply PCA to visualise the clusters in two dimensions.
- Compare models using Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Score.
- Profile clusters using descriptive statistics.

## 4. Dataset Description

The project uses the public Mall Customers CSV dataset. Common columns are CustomerID, Gender/Genre, Age, Annual Income (k$), and Spending Score (1–100). The dataset contains approximately 200 records in the commonly distributed version; verify the actual row count printed by the program before reporting it.

`CustomerID` is an identifier and is excluded from modelling. The primary features are Age, Annual Income (k$), and Spending Score (1–100). Gender is preserved in the loaded dataset but excluded from the primary numeric model to keep the feature selection transparent.

## 5. Methodology

### 5.1 Data preparation

The script checks required columns, removes duplicate rows, converts the selected feature columns to numeric, and excludes records missing any selected feature. It reports missing-value counts before modelling. The three numeric features are standardised using `StandardScaler`, so differences in units do not dominate Euclidean-distance-based clustering.

### 5.2 K-Means clustering

K-Means partitions observations into a specified number of clusters by iteratively assigning each observation to its nearest centroid and updating centroids. The Elbow Method plots inertia against `k`. Silhouette scores are also calculated for candidate values from 2 to 10 (bounded by the available record count). For the model comparison, the script selects the tested `k` with the highest silhouette score. This is a transparent selection rule, not a guarantee of a universally optimal business segmentation.

### 5.3 Agglomerative hierarchical clustering

Agglomerative clustering begins with individual observations and repeatedly merges groups according to a linkage criterion. This project uses Ward linkage and the same selected cluster count as K-Means to support a more comparable evaluation. A dendrogram is created from a sample of up to 60 observations for readability.

### 5.4 Principal Component Analysis

PCA transforms correlated numeric features into orthogonal components that capture decreasing amounts of variance. The first two components are used to display cluster assignments in two dimensions. In this project PCA is used for visualisation; the primary clustering models are fitted on the three standardised original features.

### 5.5 Evaluation measures

- **Silhouette Score:** assesses cohesion and separation; values closer to 1 generally indicate better-separated clusters.
- **Davies–Bouldin Index:** measures similarity between clusters; lower values are generally preferable.
- **Calinski–Harabasz Score:** compares between-cluster dispersion with within-cluster dispersion; higher values are generally preferable.

These metrics are internal validation measures. They should be interpreted together with plots and cluster profiles, rather than used as the sole basis for business decisions.

## 6. Implementation and Tools

- Python
- pandas and NumPy for data handling
- scikit-learn for standardisation, clustering, PCA, and evaluation
- SciPy for hierarchical linkage and dendrograms
- Matplotlib for plots

Run the project with `python main.py` after installing `requirements.txt`. The source code and generated output files are documented in `README.md`.

## 7. Results and Discussion

**Complete this section after running the program.** Copy the actual values from `outputs/clustering_metrics.csv` and inspect `outputs/cluster_profiles.csv`.

- Number of records retained after cleaning: **[enter actual value]**
- Selected cluster count (highest tested silhouette score): **[enter actual value]**
- K-Means Silhouette Score: **[enter actual value]**
- K-Means Davies–Bouldin Index: **[enter actual value]**
- K-Means Calinski–Harabasz Score: **[enter actual value]**
- Hierarchical Silhouette Score: **[enter actual value]**
- Hierarchical Davies–Bouldin Index: **[enter actual value]**
- Hierarchical Calinski–Harabasz Score: **[enter actual value]**
- Variance explained by the first two principal components: **[enter actual value]**

Discuss whether the metrics favour one algorithm and whether the PCA plots show visible separation. Use the cluster profiles to describe each group in terms of average age, income, spending score, and group size. Avoid naming groups “premium,” “low-value,” or similar unless the descriptive statistics support that interpretation.

## 8. Limitations

The dataset is small and educational rather than a verified sample of a specific mall's current customers. The meaning and construction of the spending score are not fully documented here. Results may vary with features, preprocessing, algorithm settings, and cluster-count selection. Clustering reveals associations and structure, not causal relationships.

## 9. Conclusion

This project demonstrates a complete unsupervised data mining workflow: inspection, feature preparation, scaling, clustering, dimensionality reduction for visualisation, internal evaluation, and cluster profiling. The final conclusion should be written after reviewing the actual outputs, explaining which method appears more suitable on this dataset and acknowledging the limitations of the data.

## 10. References

1. Mall Customers dataset repository: https://github.com/hannazan/mall-customer-segmentation/blob/main/data/Mall_Customers.csv
2. scikit-learn documentation: https://scikit-learn.org/stable/
3. SciPy hierarchical clustering documentation: https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
4. pandas documentation: https://pandas.pydata.org/docs/
5. Matplotlib documentation: https://matplotlib.org/stable/
