from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("MatrixMultiplication").getOrCreate()
sc = spark.sparkContext

# Example matrices
matrix_A = [
    (0, 0, 4), (0, 1, 6), (0, 2, 8),
    (1, 0, 5), (1, 1, 5), (1, 2, 4)
]

matrix_B = [
    (0, 0, 7), (0, 1, 8),
    (1, 0, 9), (1, 1, 10),
    (2, 0, 11), (2, 1, 12)
]

# Convert matrices into RDDs
rdd_A = sc.parallelize(matrix_A)  # (row, col, value)
rdd_B = sc.parallelize(matrix_B)  # (row, col, value)

# Map phase: Convert matrix entries into (key, value) pairs
mapped_A = rdd_A.map(lambda x: (x[1], (x[0], x[2])))  # Keyed by column of A
mapped_B = rdd_B.map(lambda x: (x[0], (x[1], x[2])))  # Keyed by row of B

# Join on common key (column index of A and row index of B)
joined = mapped_A.join(mapped_B)  # (col, ((row_A, val_A), (col_B, val_B)))

# Compute partial products
partial_products = joined.map(lambda x: ((x[1][0][0], x[1][1][0]), x[1][0][1] * x[1][1][1]))

# Reduce phase: Sum partial products for each (row, col) position
result = partial_products.reduceByKey(lambda x, y: x + y)

# Collect and print results
output = result.collect()
for ((row, col), value) in sorted(output):
    print(f"({row}, {col}) -> {value}")

# Stop Spark session
spark.stop()









from pyspark.sql import SparkSession

def get_matrix_input(name):
    print(f"\nEnter dimensions of matrix {name} (rows cols):")
    rows, cols = map(int, input().split())
    print(f"Enter the elements of matrix {name} row by row (space-separated):")
    matrix = []
    for i in range(rows):
        row_values = list(map(int, input().split()))
        for j in range(cols):
            matrix.append((i, j, row_values[j]))
    return matrix, rows, cols

# Initialize Spark session
spark = SparkSession.builder.appName("MatrixMultiplication").getOrCreate()
sc = spark.sparkContext

# Input matrices
matrix_A, rows_A, cols_A = get_matrix_input("A")
matrix_B, rows_B, cols_B = get_matrix_input("B")

# Validate multiplication condition
if cols_A != rows_B:
    print("Matrix multiplication not possible: Columns of A must equal rows of B.")
    spark.stop()
    exit()

# Convert matrices into RDDs
rdd_A = sc.parallelize(matrix_A)  # (row, col, value)
rdd_B = sc.parallelize(matrix_B)  # (row, col, value)

# Map phase: Convert matrix entries into (key, value) pairs
mapped_A = rdd_A.map(lambda x: (x[1], (x[0], x[2])))  # Keyed by column of A
mapped_B = rdd_B.map(lambda x: (x[0], (x[1], x[2])))  # Keyed by row of B

# Join on common key (column index of A and row index of B)
joined = mapped_A.join(mapped_B)

# Compute partial products
partial_products = joined.map(lambda x: ((x[1][0][0], x[1][1][0]), x[1][0][1] * x[1][1][1]))

# Reduce phase: Sum partial products for each (row, col) position
result = partial_products.reduceByKey(lambda x, y: x + y)

# Collect and print results
output = result.collect()
print("\nResultant Matrix C (A x B):")
for ((row, col), value) in sorted(output):
    print(f"({row}, {col}) -> {value}")

# Stop Spark session
spark.stop()

