from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Create Spark Session
spark = SparkSession.builder \
    .appName("MallCustomerClusterAnalysis") \
    .getOrCreate()

# Step 2: Load dataset
df = spark.read.csv("Mall_Customers.csv", header=True, inferSchema=True)

# Step 3: Feature Engineering
vec_assembler = VectorAssembler(
    inputCols=["Annual Income (k$)", "Spending Score (1-100)"],
    outputCol="features_vec"
)
df_vec = vec_assembler.transform(df)

scaler = StandardScaler(
    inputCol="features_vec",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)
scaler_model = scaler.fit(df_vec)
df_scaled = scaler_model.transform(df_vec)

# Step 4: Ask user for clustering choice
print("\nChoose Clustering Algorithm:")
print("1. K-Means Clustering")
print("2. Agglomerative Clustering")
choice = input("Enter your choice (1 or 2): ")

k = int(input("Enter the number of clusters: "))

if choice == "1":
    # ========== K-MEANS ==========
    kmeans = KMeans(k=k, seed=1, featuresCol="scaled_features", predictionCol="kmeans_cluster")
    kmeans_model = kmeans.fit(df_scaled)
    kmeans_result = kmeans_model.transform(df_scaled)

    print("\n=== K-Means Clustering Results ===")
    kmeans_result.select("CustomerID", "Annual Income (k$)", "Spending Score (1-100)", "kmeans_cluster").show()

    # Visualization
    kmeans_pd = kmeans_result.select("Annual Income (k$)", "Spending Score (1-100)", "kmeans_cluster").toPandas()
    sns.scatterplot(
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="kmeans_cluster",
        data=kmeans_pd,
        palette="coolwarm"
    )
    plt.title("K-Means Clustering")
    plt.show()

elif choice == "2":
    # ========== AGGLOMERATIVE ==========
    pandas_df = df_scaled.select("Annual Income (k$)", "Spending Score (1-100)").toPandas()
    scaled_data = pandas_df.values

    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(scaled_data)
    pandas_df["agg_cluster"] = agg_labels

    print("\n=== Agglomerative Clustering Results ===")
    print(pandas_df.head())

    # Visualization
    sns.scatterplot(
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="agg_cluster",
        data=pandas_df,
        palette="viridis"
    )
    plt.title("Agglomerative Clustering")
    plt.show()

else:
    print("Invalid choice!")

# Step 5: Stop Spark
spark.stop()
