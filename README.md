# Task 15 - K-Means Clustering | AI/ML Phase 1

## Project Overview

This project implements K-Means Clustering on a real-world customer segmentation dataset to identify meaningful customer groups and generate business recommendations.

## Objective

- Apply unsupervised learning using K-Means clustering.
- Discover customer segments.
- Evaluate cluster quality using Silhouette Score and Elbow Method.
- Generate business insights and recommendations.

---

## Dataset

Dataset Used: Mall Customers Dataset

Features:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

---

## Project Structure

Task15_AIML/
│
├── data/
│ └── Mall_Customers.csv
│
├── screenshots/
│ ├── elbow_method.png
│ └── clusters.png
│
├── main.py
├── clustered_customers.csv
├── requirements.txt
├── .gitignore
└── README.md

---

## Workflow

### 1. Data Loading

Dataset imported using Pandas.

### 2. Feature Selection

Selected features:

- Annual Income
- Spending Score

### 3. Data Preprocessing

Performed feature scaling using StandardScaler.

### 4. Elbow Method

Used to determine the optimal number of clusters.

### 5. K-Means Clustering

Applied K-Means algorithm with K=5.

### 6. Evaluation

Calculated Silhouette Score to validate clustering performance.

### 7. Cluster Profiling

Created customer segments and assigned business names.

### 8. Stability Testing

Validated cluster consistency using multiple random seeds.

---

## Customer Segments

- Budget Customers
- Premium Customers
- Young Spenders
- Regular Customers
- High Income Low Spend Customers

---

## Business Recommendations

| Segment               | Recommendation         |
| --------------------- | ---------------------- |
| Budget Customers      | Discount campaigns     |
| Premium Customers     | VIP memberships        |
| Young Spenders        | Social media marketing |
| Regular Customers     | Loyalty programs       |
| High Income Low Spend | Personalized offers    |

---

## Results

- K-Means clustering successfully implemented.
- Cluster quality validated using Silhouette Score.
- Business insights generated from customer segmentation.

---

## Future Improvements

- DBSCAN implementation
- Hierarchical clustering comparison
- Interactive dashboard
- Model deployment using FastAPI

---

## Author

Abhishek Kakarwal
AI/ML Developer Internship
