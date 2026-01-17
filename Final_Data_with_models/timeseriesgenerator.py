import pandas as pd
from datetime import datetime, timedelta

def add_timestamp_column(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Starting timestamp
    start_time = datetime(2025, 10, 25, 9, 0)

    # Generate timestamp values (5-minute interval)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(len(df))]

    # Insert timestamp column at the beginning
    df.insert(0, "timestamp", timestamps)

    # Save to a new file
    output_file = "Combined-Dataset-with-timestamps.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Timestamp column added successfully. Saved as {output_file}")

if __name__ == "__main__":
    add_timestamp_column("Combined-Dataset.xlsx")
