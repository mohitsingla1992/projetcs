from google.cloud import bigquery

# Create a BigQuery client
client = bigquery.Client()

# Set the table ID
table_id = "bigquery-public-data.austin_311.311_service_requests"

# Construct a query to fetch the first 5 rows
query = f"SELECT * FROM `{table_id}` LIMIT 5"

# Execute the query
query_job = client.query(query)  # API request
rows = query_job.result()  # Waits for query to finish

# Print the first 5 rows
for row in rows[:5]:
    print(row)
