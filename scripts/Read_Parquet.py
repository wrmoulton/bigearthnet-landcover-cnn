import pandas as pd

# Load the metadata parquet file
metadata_path = "data/metadata.parquet"
md = pd.read_parquet(metadata_path, engine="pyarrow")

# Filter for Portugal patches only and extract year from patch_id
md_portugal = md[md["country"] == "Portugal"].copy()
md_portugal["year"] = md_portugal["patch_id"].apply(lambda pid: int(pid.split("_")[2][:4]))

# Count number of patches per year
patch_counts_by_year = md_portugal["year"].value_counts().sort_index()
print(patch_counts_by_year)
