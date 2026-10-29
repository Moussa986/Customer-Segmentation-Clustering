# Viva Preparation: Customer Segmentation Using Clustering

1. **What is customer segmentation?**  
   It is the process of grouping customers with similar characteristics or behaviours.

2. **Why is this an unsupervised learning problem?**  
   The dataset has no predefined target class. The algorithms discover groupings from feature similarity.

3. **What is the difference between classification and clustering?**  
   Classification learns from labelled examples to predict known classes; clustering groups unlabelled observations by similarity.

4. **Why is CustomerID excluded?**  
   It is an identifier and does not represent a behavioural or demographic measurement. Including it could introduce meaningless distance information.

5. **Why standardise the features?**  
   Age, income, and spending score have different scales. Standardisation places them on comparable scales for distance-based methods.

6. **How does K-Means work?**  
   It assigns points to the nearest centroid and repeatedly recalculates centroids until assignments or centroid positions stabilise.

7. **What is inertia in the Elbow Method?**  
   It is the sum of squared distances from each observation to its assigned cluster centroid. It generally decreases as the number of clusters increases.

8. **What is the Silhouette Score?**  
   It measures how similar a point is to its own cluster compared with other clusters. It ranges from -1 to 1; larger values generally indicate better cohesion and separation.

9. **What is hierarchical clustering?**  
   It builds a hierarchy of nested clusters. Agglomerative clustering starts with individual observations and merges them step by step.

10. **What does Ward linkage do?**  
    It chooses merges that minimise the increase in within-cluster variance.

11. **What is a dendrogram?**  
    A tree diagram showing the sequence and linkage distances at which observations or clusters merge.

12. **What is PCA?**  
    Principal Component Analysis transforms features into orthogonal components ordered by the amount of variance they explain.

13. **Why use PCA in this project?**  
    The first two components provide a two-dimensional view of cluster assignments from the three-feature data. PCA is for visualisation here, not the main model input.

14. **What is the Davies–Bouldin Index?**  
    An internal clustering metric based on within-cluster scatter and between-cluster separation; lower is generally better.

15. **What is the Calinski–Harabasz Score?**  
    A ratio of between-cluster dispersion to within-cluster dispersion; higher is generally better.

16. **Can clustering prove a customer group is profitable?**  
    No. The clusters describe patterns in the available data. Profitability would require additional business outcomes and validation.

17. **Why compare more than one metric?**  
    Each metric captures a different aspect of cluster structure, so considering several measures reduces reliance on a single score.

18. **What are the limitations of this project?**  
    The dataset is small and educational, features are limited, the spending score's construction is not documented here, and clusters depend on feature selection and algorithm settings.

19. **How can the project be improved?**  
    Validate segments on a larger real-world dataset, include purchase frequency or transaction value, test alternative features and clustering methods, and evaluate whether segments support useful business decisions.

20. **What should you say about your results?**  
    Explain the values and patterns actually produced by your run. Do not memorise or claim numerical results without checking the generated files.
