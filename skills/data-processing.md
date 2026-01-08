# Data Processing

## Pandas Operations
```python
import pandas as pd

# Reading
df = pd.read_csv("data.csv", dtype={"id": str})  # Specify dtypes
df = pd.read_json("data.json", lines=True)  # For JSONL

# Filtering
filtered = df[df["column"] > 100]
filtered = df.query("column > 100 and status == 'active'")

# Aggregation
grouped = df.groupby("category").agg({
    "value": ["sum", "mean", "count"],
    "date": "max"
})

# Memory optimization
df = pd.read_csv("large.csv", chunksize=10000)
for chunk in df:
    process(chunk)
```

## Data Cleaning
- Handle missing values: `df.fillna()` or `df.dropna()`
- Remove duplicates: `df.drop_duplicates()`
- Type conversion: `df.astype()`
- String cleaning: `df["col"].str.strip().str.lower()`

## Common Pitfalls
- SettingWithCopyWarning: Use `.loc[]` for assignment
- Memory issues with large files: Use chunking or Dask
- Incorrect dtypes: Specify on read or convert after
- Timezone issues: Always use `pd.to_datetime(utc=True)`

## Performance Tips
- Use `df.itertuples()` not `df.iterrows()`
- Prefer vectorized operations over loops
- Use categorical dtype for low-cardinality columns
- Consider Polars for large datasets (10x faster)

## Polars (Alternative)
```python
import polars as pl

df = pl.read_csv("data.csv")
result = (
    df.filter(pl.col("value") > 100)
    .group_by("category")
    .agg(pl.col("value").sum())
)
```

## Verification
- Check shape before/after operations
- Validate no unexpected nulls
- Spot check random samples
