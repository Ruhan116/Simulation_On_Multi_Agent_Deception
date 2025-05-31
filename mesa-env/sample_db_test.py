import sqlite3

db_path = "benchmark_data.db"  # Change if your DB file has a different name

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

for table in tables:
    print(f"\n--- {table} ---")
    cursor.execute(f"SELECT * FROM {table}")
    rows = cursor.fetchall()
    # Print column names
    col_names = [description[0] for description in cursor.description]
    print(col_names)
    for row in rows:
        print(row)

conn.close()