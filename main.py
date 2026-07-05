import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Program Started...")

# Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Feature Selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData scaling completed successfully.")

# ====================================
# Elbow Method
# ====================================
print("\nRunning Elbow Method...")

wcss = []

for i in range(1, 11):
    km = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)

plt.savefig("screenshots/elbow_method.png")
plt.close()

print("Elbow Method Completed")

# ====================================
# KMeans
# ====================================
print("\nRunning KMeans...")

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ====================================
# Silhouette Score
# ====================================
score = silhouette_score(X_scaled, clusters)

print("\nSilhouette Score:")
print(score)

# ====================================
# Cluster Plot
# ====================================
plt.figure(figsize=(8,6))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=clusters
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker='X',
    s=300
)

plt.title("KMeans Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.savefig("screenshots/clusters.png")
plt.close()

# ====================================
# Cluster Names
# ====================================
cluster_names = {
    0: "Budget Customers",
    1: "Premium Customers",
    2: "Young Spenders",
    3: "Regular Customers",
    4: "High Income Low Spend"
}

df["Cluster_Name"] = df["Cluster"].map(cluster_names)

print("\nCluster Counts:")
print(df["Cluster_Name"].value_counts())

# ====================================
# Business Recommendations
# ====================================
recommendations = {
    "Budget Customers":
        "Provide discount offers",

    "Premium Customers":
        "Provide VIP memberships",

    "Young Spenders":
        "Target using social media",

    "Regular Customers":
        "Provide loyalty programs",

    "High Income Low Spend":
        "Provide personalized offers"
}

print("\nBusiness Recommendations:\n")

for c, r in recommendations.items():
    print(f"{c}: {r}")

# ====================================
# Stability Test
# ====================================
print("\nStability Test:\n")

for seed in [0,42,100]:

    km = KMeans(
        n_clusters=5,
        random_state=seed,
        n_init=10
    )

    pred = km.fit_predict(X_scaled)

    print(
        f"Seed={seed}",
        "Score=",
        round(
            silhouette_score(
                X_scaled,
                pred
            ),
            3
        )
    )

# ====================================
# Cluster Profile
# ====================================
print("\nCluster Profiles:\n")

print(
    df.groupby("Cluster")
      .mean(numeric_only=True)
)

# ====================================
# Save Final CSV
# ====================================
df.to_csv(
    "clustered_customers.csv",
    index=False
)

print("\nclustered_customers.csv saved successfully")
print("\nTask15 Completed Successfully")