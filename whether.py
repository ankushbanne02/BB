from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, max

# Start Spark session
spark = SparkSession.builder.appName("Max Snowfall Finder").getOrCreate()

# Load CSV
df = spark.read.csv("weather_data_1000.csv", header=True, inferSchema=True)

# Add 'year' column
df = df.withColumn("year", year(col("date")))

# Filter for year 2022
df_filtered = df.filter(col("year") == 2022)

# Find max snowfall
max_snowfall = df_filtered.agg(max("snowfall").alias("max_snow")).collect()[0]["max_snow"]

# Get rows with max snowfall
max_snowfall_df = df_filtered.filter(col("snowfall") == max_snowfall)

# Show results
max_snowfall_df.select("station_id", "date", "snowfall").show()

# Optional: Save result
max_snowfall_df.select("station_id", "date", "snowfall").coalesce(1).write.csv("output_max_snowfall", header=True, mode="overwrite")

# Stop Spark
spark.stop()