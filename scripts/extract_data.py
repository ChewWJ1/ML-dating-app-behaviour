import pandas as pd
import json
import joblib

dataset_path = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\dating_app_behavior_dataset_extended1.csv'
df = pd.read_csv(dataset_path)

stats = {}

# 1. EDA Statistics
categorical_cols = ['match_outcome', 'gender', 'location_type', 'income_bracket', 
                   'swipe_time_of_day', 'sexual_orientation', 'body_type', 'relationship_intent']

for col in categorical_cols:
    stats[col] = df[col].value_counts().to_dict()

numeric_cols = ['app_usage_time_min', 'swipe_right_ratio', 'likes_received', 'mutual_matches',
               'bio_length', 'message_sent_count', 'emoji_usage_rate', 'profile_pics_count', 'age']

for col in numeric_cols:
    stats[col] = {
        'mean': float(df[col].mean()),
        'min': float(df[col].min()),
        'max': float(df[col].max())
    }

# Binary target distribution
if 'match_outcome' in df.columns:
    # Assuming positive classes are: Mutual Match, Instant Match, Date Happened, Relationship Formed
    pos_classes = ['Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed']
    df['target'] = df['match_outcome'].apply(lambda x: 1 if x in pos_classes else 0)
    stats['target_distribution'] = df['target'].value_counts().to_dict()

# Interest tags
all_tags = []
for tags in df['interest_tags'].dropna():
    all_tags.extend([tag.strip() for tag in tags.split(',')])
stats['top_interest_tags'] = pd.Series(all_tags).value_counts().head(10).to_dict()

stats['total_rows'] = len(df)
stats['total_columns'] = len(df.columns)

# Write to JSON
with open('eda_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("EDA stats successfully saved to eda_stats.json")

# Also try to extract from joblib
try:
    cv_results = joblib.load('cv_results.joblib')
    with open('cv_stats.json', 'w') as f:
        # Convert any numpy types to python native for JSON serialization if necessary
        # We'll just convert to string for simplicity to avoid TypeErrors
        def default_serializer(obj):
            return str(obj)
        json.dump(cv_results, f, default=default_serializer, indent=4)
    print("cv_results extracted")
except Exception as e:
    print(f"Could not load cv_results: {e}")
