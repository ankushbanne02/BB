from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("IrisPCAandLogisticRegression").getOrCreate()

# Load the Iris dataset (replace with your actual file path or use built-in dataset)
df = spark.read.csv("iris.csv", header=True, inferSchema=True)

# Step 1: Show the first few rows to check the data
df.show(5)

# Step 2: Check the schema to confirm column names and types
df.printSchema()

# Step 3: Calculate and visualize the correlation matrix
# Convert to Pandas for visualization
pandas_df = df.select("sepal_length", "sepal_width", "petal_length", "petal_width").toPandas()

# Calculate the correlation matrix
corr_matrix = pandas_df.corr()

# Visualize the correlation matrix with a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1, square=True)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Step 4: Perform Dimensionality Reduction (PCA)
# Assemble the features into a single vector column
assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features"
)
df = assembler.transform(df)

# Standardize the features (important for PCA)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model = scaler.fit(df)
scaled_df = scaler_model.transform(df)

# Apply PCA to reduce the dimensionality to 2 components
pca = PCA(k=2, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_model = pca.fit(scaled_df)
pca_result = pca_model.transform(scaled_df)

# Show the PCA results
pca_result.select("pcaFeatures").show(5, truncate=False)

# Step 5: Logistic Regression for classification
# Use StringIndexer to convert the species column into numeric labels
indexer = StringIndexer(inputCol="species", outputCol="label")
indexed_df = indexer.fit(pca_result).transform(pca_result)

# Split the dataset into training and testing sets
train_df, test_df = indexed_df.randomSplit([0.8, 0.2], seed=1234)

# Train the logistic regression model
lr = LogisticRegression(featuresCol="pcaFeatures", labelCol="label", maxIter=10)
lr_model = lr.fit(train_df)

# Step 6: Make predictions on the test data
predictions = lr_model.transform(test_df)

# Step 7: Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Show the predictions
predictions.select("species", "prediction", "probability").show(5)

# Step 9: Visualize PCA results (2D plot)
pandas_df = indexed_df.select("pcaFeatures", "species", "label").toPandas()
pandas_df[['PC1', 'PC2']] = pd.DataFrame(pandas_df['pcaFeatures'].to_list(), index=pandas_df.index)

# Plot the PCA results with species as hue
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="species", data=pandas_df, palette="Set1")
plt.title("PCA of Iris Dataset (2 Components)")
plt.show()
