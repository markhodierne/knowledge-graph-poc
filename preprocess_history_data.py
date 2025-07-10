import pandas as pd

# Load CSV files
source_df = pd.read_csv("cat_data_wtwn.csv")
target_df = pd.read_csv("cat_data_history_with_descriptions.csv")

# Rename two columns
source_df = source_df.rename(columns={
    'unit_id': 'unitId',
    'why_this_why_now': 'whyThisWhyNow'
})

# Ensure consistent types for joining
source_df['unitId'] = source_df['unitId'].astype(str)
target_df['unitId'] = target_df['unitId'].astype(str)

# Drop duplicate lessonId rows in source_df, keeping the first occurrence
deduped_source = source_df[['unitId', 'whyThisWhyNow']].drop_duplicates(subset='unitId', keep='first')

# Merge on lessonId
target_df = target_df.merge(
    deduped_source,
    on='unitId',
    how='left'
)

# --- VALIDATION CHECK ---

missing_outcomes = target_df['whyThisWhyNow'].isna().sum()
print("✅ All rows have whyThisWhyNow." if missing_outcomes == 0
      else f"❌ {missing_outcomes} row(s) missing whyThisWhyNow.")

# --- SAVE OUTPUT ---

target_df.to_csv("cat_data_history_with_enrichments.csv", index=False)
print("📁 Enriched file saved as 'cat_data_history_with_enrichments.csv'")
