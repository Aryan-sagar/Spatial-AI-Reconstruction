"""Read-only diagnostic: where do non-finite odometry values come from?  usage: python scripts/diagnose_odometry.py <scan_dir>"""
import sys
from pathlib import Path
import pandas as pd

p = Path(sys.argv[1]) / "odometry.csv"
df = pd.read_csv(p, skipinitialspace=True)
df.columns = df.columns.str.strip()
print("rows:", len(df), "\ncolumns:", list(df.columns))
num = df.apply(pd.to_numeric, errors="coerce")
print("\nnon-finite count per column:\n", (~num.apply(lambda c: c.map(pd.notna))).sum().to_string())
print("\ndtypes:\n", df.dtypes.to_string())
print("\nfirst 3 rows:\n", df.head(3).to_string())
