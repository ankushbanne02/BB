from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create Spark Session
spark = SparkSession.builder \
    .appName("MultivariateAnalysisBigData") \
    .getOrCreate()

# Step 2: Load dataset
df = spark.read.csv("Mall_Customers.csv", header=True, inferSchema=True)

# Step 3: Feature Engineering (select columns for analysis)
df_selected = df.select("Age", "Annual Income (k$)", "Spending Score (1-100)")

# Step 4: Feature Vectorization
vec_assembler = VectorAssembler(inputCols=["Age", "Annual Income (k$)", "Spending Score (1-100)"], outputCol="features_vec")
df_vec = vec_assembler.transform(df_selected)

# Step 5: Scaling the features (Standardization)
scaler = StandardScaler(inputCol="features_vec", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df_vec)
df_scaled = scaler_model.transform(df_vec)

# Step 6: Principal Component Analysis (PCA) for dimensionality reduction
from pyspark.ml.feature import PCA

pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

# Step 7: Visualizing PCA results (2D plot)
pca_df = df_pca.select("pca_features").rdd.map(lambda row: row[0].toArray()).collect()
pca_df = [x.tolist() for x in pca_df]

pca_result = pd.DataFrame(pca_df, columns=["PC1", "PC2"])
sns.scatterplot(x="PC1", y="PC2", data=pca_result, palette="viridis")
plt.title("PCA - Mall Customers Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Step 8: Correlation Analysis
# Compute correlation matrix
vector_col = "scaled_features"
matrix = Correlation.corr(df_scaled, vector_col, method="pearson").head()[0]
corr_matrix = matrix.toArray()

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", xticklabels=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
            yticklabels=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
plt.title("Correlation Matrix")
plt.show()

# Step 9: K-Means Clustering (optional for clustering in multivariate space)
kmeans = KMeans(k=3, seed=1, featuresCol="scaled_features", predictionCol="cluster")
kmeans_model = kmeans.fit(df_scaled)
kmeans_result = kmeans_model.transform(df_scaled)

# Show K-Means result (optional)
print("\n=== K-Means Clustering Results ===")
kmeans_result.select("Age", "Annual Income (k$)", "Spending Score (1-100)", "cluster").show()

# Step 10: Stop Spark Session
spark.stop()
