import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

mod_cell17 = False
mod_cell23 = False

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_str = "".join(source)
        
        # 1. Update Cell 17 (Causal cell preprocessing)
        if "from sklearn.preprocessing import OrdinalEncoder, RobustScaler, MultiLabelBinarizer" in source_str and "df_processed = df_temp" in source_str:
            new_source = []
            for line in source:
                if "'likes_received', 'mutual_matches', 'profile_pics_count', 'bio_length'," in line and "'message_sent_count', 'emoji_usage_rate', 'last_active_hour']" in source_str:
                    # Let's replace the numeric_cols list definition in Cell 17
                    if "numeric_cols = ['age'," in line or "'likes_received', 'mutual_matches'" in line:
                        pass # skip lines to reconstruct
                new_source = source # fallback
            
            # Let's do a simple string replacement on the cell source list:
            joined = "".join(source)
            target_str = """numeric_cols = ['age', 'height_cm', 'weight_kg', 'app_usage_time_min', 'swipe_right_ratio',
                'likes_received', 'mutual_matches', 'profile_pics_count', 'bio_length',
                'message_sent_count', 'emoji_usage_rate', 'last_active_hour']"""
            
            replacement_str = """numeric_cols = ['age', 'height_cm', 'weight_kg', 'app_usage_time_min', 'swipe_right_ratio',
                'likes_received', 'mutual_matches', 'profile_pics_count', 'bio_length',
                'message_sent_count', 'emoji_usage_rate', 'last_active_hour',
                'engagement_score', 'profile_completeness', 'activity_intensity', 'selectivity_ratio',
                'likes_received_log', 'message_sent_count_log', 'bio_length_log', 'app_usage_time_min_log']"""
            
            if target_str in joined:
                joined = joined.replace(target_str, replacement_str)
                cell['source'] = [l + '\n' for l in joined.split('\n')][:-1]
                mod_cell17 = True
                print("Modified Cell 17 scaling!")
        
        # 2. Update Cell 23 (RobustScaler cell)
        if "from sklearn.preprocessing import RobustScaler" in source_str and "df[numeric_cols] = scaler.fit_transform(df[numeric_cols])" in source_str:
            joined = "".join(source)
            target_str = """    'likes_received', 'mutual_matches',
    'profile_pics_count', 'bio_length',
    'message_sent_count', 'emoji_usage_rate',
    'last_active_hour'"""
            
            replacement_str = """    'likes_received', 'mutual_matches',
    'profile_pics_count', 'bio_length',
    'message_sent_count', 'emoji_usage_rate',
    'last_active_hour',
    'engagement_score', 'profile_completeness', 'activity_intensity', 'selectivity_ratio',
    'likes_received_log', 'message_sent_count_log', 'bio_length_log', 'app_usage_time_min_log'"""
            
            if target_str in joined:
                joined = joined.replace(target_str, replacement_str)
                cell['source'] = [l + '\n' for l in joined.split('\n')][:-1]
                mod_cell23 = True
                print("Modified Cell 23 scaling!")

if mod_cell17 or mod_cell23:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully saved notebook changes!")
else:
    print("No changes were made!")
