# Customer Clustering Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Clustering-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An unsupervised machine learning project that segments customers using age, annual income, and spending score. The project compares clustering methods and visualizes customer groups for business analysis.

## Project Highlights

- Prepared customer profile data for clustering analysis.
- Standardized numerical features before model fitting.
- Compared K-Means, Agglomerative Clustering, and DBSCAN.
- Used silhouette score and cluster sizes to compare model quality.
- Added a reproducible script that exports customer segment labels.

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

## Project Structure

```text
customer-clustering-project/
|-- data/
|   `-- customers.csv
|-- notebooks/
|   `-- clustering_project_aatman.ipynb
|-- src/
|   `-- cluster_customers.py
|-- requirements.txt
`-- README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Aatmanium/customer-clustering-project.git
cd customer-clustering-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the clustering script:

```bash
python src/cluster_customers.py
```

Open the notebook:

```bash
jupyter notebook notebooks/clustering_project_aatman.ipynb
```

## Methodology

1. Load customer data from `data/customers.csv`.
2. Select age, annual income, and spending score as clustering features.
3. Impute missing values with the median.
4. Standardize features to place them on the same scale.
5. Fit K-Means, Agglomerative Clustering, and DBSCAN models.
6. Compare model performance with silhouette scores.
7. Export final customer segment labels to `outputs/customer_segments.csv`.

## Business Value

Customer segmentation can help teams understand shopper behavior, design targeted campaigns, and identify groups such as high-income low-spending customers or high-value frequent buyers.

## Future Improvements

- Add richer customer features such as purchase frequency and recency.
- Build an interactive dashboard for exploring segments.
- Tune DBSCAN parameters with nearest-neighbor distance plots.
- Add model interpretation notes for each cluster profile.

## Author

**Aatmanium**  
Applied AI Student | Machine Learning Enthusiast | Python Developer
