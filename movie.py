from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

# Step 1: Create a Spark session
spark = SparkSession.builder \
    .appName("MovieRatingsAverage") \
    .getOrCreate()

# Step 2: Load the CSV file (Update the path as needed)
df = spark.read.csv("movies_ratings_100.csv", header=True, inferSchema=True)

# Step 3: Show schema (optional)
df.printSchema()

# Step 4: Select relevant columns
df_selected = df.select("movie_id", "rating")

# Step 5: Perform MapReduce logic
# Group by movie_id and calculate average rating
avg_ratings = df_selected.groupBy("movie_id").agg(avg("rating").alias("average_rating"))

# Step 6: Show results
avg_ratings.orderBy("movie_id").show()

# Step 7: Stop the Spark session
spark.stop()
