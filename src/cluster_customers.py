from pathlib import Path

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "customers.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
FEATURES = ["Age", "Annual_Income", "Spending_Score"]


def load_customer_data(data_path=DATA_PATH):
    customers = pd.read_csv(data_path)
    missing_columns = [column for column in FEATURES if column not in customers]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return customers


def prepare_features(customers):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed_features = imputer.fit_transform(customers[FEATURES])
    return scaler.fit_transform(imputed_features)


def safe_silhouette_score(features, labels):
    unique_labels = set(labels)

    if len(unique_labels) < 2:
        return None

    if unique_labels == {-1}:
        return None

    return silhouette_score(features, labels)


def run_clustering():
    customers = load_customer_data()
    scaled_features = prepare_features(customers)

    models = {
        "KMeans": KMeans(n_clusters=5, random_state=42, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=5),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    }

    scores = {}
    for name, model in models.items():
        labels = model.fit_predict(scaled_features)
        customers[f"{name}_Cluster"] = labels
        scores[name] = safe_silhouette_score(scaled_features, labels)

    return customers, scores


def main():
    clustered_customers, scores = run_clustering()

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "customer_segments.csv"
    clustered_customers.to_csv(output_path, index=False)

    print("Silhouette scores")
    for name, score in scores.items():
        value = "not available" if score is None else f"{score:.3f}"
        print(f"- {name}: {value}")

    print(f"\nSaved segmented customers to {output_path}")


if __name__ == "__main__":
    main()
