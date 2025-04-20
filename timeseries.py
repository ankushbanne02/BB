from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, unix_timestamp, hour, dayofweek, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# 1. Start Spark session
spark = SparkSession.builder \
    .appName("TimeSeriesWithPredictionAndAccuracy") \
    .getOrCreate()

# 2. Load and preprocess data
df = spark.read.csv("traffic_data_extended.csv", header=True, inferSchema=True)
df = df.withColumn("timestamp", to_timestamp("timestamp"))
df = df.filter(df["timestamp"].isNotNull() & df["traffic_volume"].isNotNull())

# 3. Feature engineering
df = df.withColumn("hour", hour("timestamp")) \
       .withColumn("day_of_week", dayofweek("timestamp")) \
       .withColumn("timestamp_unix", unix_timestamp("timestamp").cast("double")) \
       .withColumn("is_weekend", ((col("day_of_week") == 1) | (col("day_of_week") == 7)).cast("int"))  # Corrected is_weekend column

# 4. Assemble features
feature_cols = ["hour", "day_of_week", "temperature", "rainfall", "is_weekend"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("timestamp", "features", "traffic_volume")

# 5. Train model
lr = LinearRegression(featuresCol="features", labelCol="traffic_volume")
lr_model = lr.fit(data)

# 6. Evaluate model
predictions = lr_model.transform(data).select("timestamp", "traffic_volume", "prediction")

evaluator_rmse = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"📏 Model Accuracy:\n - RMSE: {rmse:.2f}\n - R² Score: {r2:.4f}")

# Plotting the results
pandas_df = predictions.toPandas()
plt.figure(figsize=(14, 6))
plt.plot(pandas_df["timestamp"], pandas_df["traffic_volume"], label="Actual Traffic Volume", alpha=0.5)
plt.plot(pandas_df["timestamp"], pandas_df["prediction"], label="Predicted Volume", color='red', alpha=0.8)
plt.title("Traffic Volume Prediction Using Time and Weather Features")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# 7. Make a sample future prediction with user input
# Collect user input for future prediction
hour_input = int(input("Hour of Day (0–23): "))
day_input = int(input("Day of Week (1=Sun, 7=Sat): "))
temp_input = float(input("Temperature (°C): "))
rain_input = float(input("Rainfall (mm): "))
weekend_input = int(input("Is Weekend? (0=No, 1=Yes): "))

# Create DataFrame with user input
user_future_df = spark.createDataFrame([(hour_input, day_input, temp_input, rain_input, weekend_input)], feature_cols)

# Transform and predict
user_future_vector = assembler.transform(user_future_df).select("features")
user_future_prediction = lr_model.transform(user_future_vector)

# Show result
predicted_value = user_future_prediction.select("prediction").collect()[0][0]
print(f"\n🚗 Predicted Traffic Volume: {predicted_value:.2f} vehicles")

# 8. Stop Spark session
spark.stop()
