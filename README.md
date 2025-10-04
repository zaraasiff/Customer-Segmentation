# Customer Segmentation Using Clustering

This project is about analyzing mall customers and grouping them into different segments using machine learning algorithms like **KMeans** and **DBSCAN**. The main goal is to understand customer behavior based on their age, gender, income, and spending score.

---

## Dataset
The dataset used is **Mall_Customers.csv**, which contains:
- CustomerID  
- Gender  
- Age  
- Annual Income (k$)  
- Spending Score (1–100)  

---

## What This Project Does
1. Loads and explores the customer data.  
2. Visualizes important patterns like age, income, and spending behavior.  
3. Uses **KMeans** to create customer clusters based on income and spending score.  
4. Compares different cluster counts using **Elbow Method** and **Silhouette Score**.  
5. Applies **DBSCAN** for density-based clustering and finds natural groupings in the data.  
6. Displays 2D cluster visualizations and relationships between features.  

---

## Steps Performed

### 1. Data Loading and Exploration
- Read the CSV file using pandas.  
- Checked shape, info, and missing values.  
- Displayed gender distribution and descriptive statistics.  

### 2. Data Visualization
- Created histograms for Age, Annual Income, and Spending Score.  
- Counted gender distribution.  
- Used scatter plots to show relationships like Income vs Spending Score and Age vs Spending Score.  

### 3. Preprocessing
- Encoded the Gender column.  
- Standardized numeric features using `StandardScaler` for clustering.  

### 4. KMeans Clustering
- Used the **Elbow Method** to find the best number of clusters.  
- Calculated **Silhouette Scores** to check cluster quality.  
- Finalized **5 clusters** for the KMeans model.  
- Visualized the clusters along with their centroids (red X markers).  

### 5. DBSCAN Clustering
- Used **k-distance graph** to find the suitable epsilon (eps) value.  
- Applied DBSCAN to detect natural clusters and outliers in the data.  

---

## Results
- KMeans created 5 distinct customer groups.  
- DBSCAN detected clusters based on data density.  
- Visualizations clearly showed how different customers group together based on income and spending score.  

---

## Tools and Libraries
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## Summary
This project helps in identifying different types of mall customers. Businesses can use these clusters to design better marketing strategies — for example, targeting high-income, high-spending customers differently from low-income, low-spending ones.

---

