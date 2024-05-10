import pyarrow.parquet as pq
import pandas as pd
import argparse

class colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'

def main(parquet_path):
    # Read Parquet file
    table = pq.read_table(parquet_path)
    ml_outputs = table.to_pandas()

    # Read metadata
    schema = table.schema
    for field in schema.names:
        print(colors.BOLD + colors.YELLOW + "Column: {0}".format(field) + colors.RESET)
        print("Unit: " + colors.CYAN + "{0}".format(schema.field(field).metadata[b'unit'].decode('utf-8')) + colors.RESET)
        print("Description: " + colors.CYAN + "{0} \n".format(schema.field(field).metadata[b'description'].decode('utf-8')) + colors.RESET)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read metadata from a Parquet file")
    parser.add_argument("parquet_path", type=str, help="Path to the Parquet file")
    args = parser.parse_args()

    main(args.parquet_path)