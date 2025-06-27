import pandas as pd
import os
from app.db import SessionLocal
from app.models.session_db import SessionDB
import numpy as np
from pandas.core.generic import NDFrame

# Load datasets
backend_dir = os.path.dirname(os.path.abspath(__file__))
activity_path = os.path.join(backend_dir, 'dataset1_final.csv')
transaction_path = os.path.join(backend_dir, 'dataset2_final.csv')
activity_df = pd.read_csv(activity_path)
transaction_df = pd.read_csv(transaction_path)

# Strip whitespace from all column names and force lowercase
activity_df.columns = activity_df.columns.str.strip().str.lower()
transaction_df.columns = transaction_df.columns.str.strip().str.lower()

# Print columns for debugging
print('activity_df columns:', activity_df.columns.tolist())
print('transaction_df columns:', transaction_df.columns.tolist())

# Standardize column names for joining
activity_df = activity_df.rename(columns={'transae': 'transaction_id'})
transaction_df = transaction_df.rename(columns={'Transaction_ID': 'transaction_id'})

# Debug: print columns if join fails
if 'transaction_id' not in activity_df.columns:
    print('activity_df columns:', activity_df.columns.tolist())
if 'transaction_id' not in transaction_df.columns:
    print('transaction_df columns:', transaction_df.columns.tolist())

# Ensure transaction_id is string in both dataframes
activity_df['transaction_id'] = activity_df['transaction_id'].astype(str)
transaction_df['transaction_id'] = transaction_df['transaction_id'].astype(str)

# Merge datasets on transaction_id
merged_df = pd.merge(activity_df, transaction_df, on='transaction_id', how='left')

# Convert eventtimestamp to datetime
merged_df['eventtimestamp'] = pd.to_datetime(merged_df['eventtimestamp'], errors='coerce')

# Sort by user and timestamp
merged_df = merged_df.sort_values(['user_pseudo_id', 'eventtimestamp'])

# Construct sessions: assign a session_id for each user based on time gap (e.g., 30 min)
SESSION_GAP = pd.Timedelta(minutes=30)
merged_df['prev_event_time'] = merged_df.groupby('user_pseudo_id')['eventtimestamp'].shift(1)
merged_df['time_diff'] = merged_df['eventtimestamp'] - merged_df['prev_event_time']
merged_df['new_session'] = (merged_df['time_diff'] > SESSION_GAP) | (merged_df['time_diff'].isna())
merged_df['session_id'] = merged_df.groupby('user_pseudo_id')['new_session'].cumsum()

# Categorize sessions by engagement type (example: based on event_name)
def categorize_engagement(events):
    if 'purchase' in events.values:
        return 'high'
    elif 'add_to_cart' in events.values:
        return 'medium'
    else:
        return 'low'

session_engagement = merged_df.groupby(['user_pseudo_id', 'session_id'])['event_name'].apply(categorize_engagement).reset_index()
session_engagement = session_engagement.rename(columns={'event_name': 'engagement_type'})

# Merge engagement type back to merged_df
merged_df = pd.merge(merged_df, session_engagement, on=['user_pseudo_id', 'session_id'], how='left')

# User Segmentation

def get_income_bracket(income):
    try:
        income = float(income)
        if income < 30000:
            return 'low'
        elif income < 70000:
            return 'medium'
        else:
            return 'high'
    except Exception:
        return 'unknown'

def classify_source(source):
    if not isinstance(source, str):
        return 'unknown'
    source = source.lower()
    if any(s in source for s in ['facebook', 'instagram', 'twitter', 'linkedin', 'social']):
        return 'social'
    elif any(s in source for s in ['ad', 'cpc', 'ppc', 'paid', 'googleads', 'bingads']):
        return 'paid'
    elif any(s in source for s in ['organic', 'search', 'seo']):
        return 'organic'
    else:
        return 'other'

def get_user_segments(df):
    segments = []
    for user, group in df.groupby('user_pseudo_id'):
        events = group['event_name'].tolist()
        purchases = events.count('purchase')
        add_to_cart = events.count('add_to_cart')
        sessions = group['session_id'].nunique()
        engagement = group['engagement_type'].mode()[0] if not group['engagement_type'].isna().all() else 'unknown'

        if purchases > 1:
            segment = 'repeat_purchaser'
        elif add_to_cart > 0 and purchases == 0:
            segment = 'cart_abandoner'
        elif sessions > 3:
            segment = 'frequent_viewer'
        else:
            segment = 'other'

        # Demographics
        gender = group['gender'].mode()[0] if 'gender' in group.columns and not group['gender'].isna().all() else 'unknown'
        age = group['age'].mode()[0] if 'age' in group.columns and not group['age'].isna().all() else 'unknown'
        city = group['city'].mode()[0] if 'city' in group.columns and not group['city'].isna().all() else 'unknown'
        state = group['state'].mode()[0] if 'state' in group.columns and not group['state'].isna().all() else 'unknown'
        country = group['country'].mode()[0] if 'country' in group.columns and not group['country'].isna().all() else 'unknown'
        source = group['source'].mode()[0] if 'source' in group.columns and not group['source'].isna().all() else 'unknown'
        income = group['income'].mode()[0] if 'income' in group.columns and not group['income'].isna().all() else 'unknown'
        income_bracket = get_income_bracket(income)
        source_type = classify_source(source)

        segments.append({
            'user_pseudo_id': user,
            'segment': segment,
            'engagement_type': engagement,
            'gender': gender,
            'age': age,
            'income': income,
            'income_bracket': income_bracket,
            'city': city,
            'state': state,
            'country': country,
            'source': source,
            'source_type': source_type
        })
    return pd.DataFrame(segments)

user_segments_df = get_user_segments(merged_df)
user_segments_df.to_csv('user_segments.csv', index=False)
print('User segmentation complete. Output saved to user_segments.csv')

# --- Cold Start Strategies ---

# 1. Default Category Trends by Region (Country)
popular_by_country = merged_df.groupby('country')['event_name'].value_counts().groupby(level=0).idxmax()
popular_by_country = popular_by_country.apply(lambda x: x[1]).reset_index()
popular_by_country.columns = ['country', 'top_event']
popular_by_country.to_csv('popular_by_country.csv', index=False)
print('Popular events by country saved to popular_by_country.csv')

# 2. Profile Similarity-Based Logic (Demographic Filtering)
popular_by_demo = merged_df.groupby(['age', 'gender'])['event_name'].value_counts().groupby(level=[0,1]).idxmax()
popular_by_demo = popular_by_demo.apply(lambda x: x[2]).reset_index()
popular_by_demo.columns = ['age', 'gender', 'top_event']
popular_by_demo.to_csv('popular_by_demo.csv', index=False)
print('Popular events by demographic saved to popular_by_demo.csv')

# 3. Similar Users (Clustering)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

features = user_segments_df[['age', 'gender', 'city', 'state', 'country']].fillna('unknown')
for col in features.columns:
    features[col] = LabelEncoder().fit_transform(features[col].astype(str))

kmeans = KMeans(n_clusters=5, random_state=42)
user_segments_df['cluster'] = kmeans.fit_predict(features)

merged_with_cluster = merged_df.merge(user_segments_df[['user_pseudo_id', 'cluster']], on='user_pseudo_id', how='left')
popular_by_cluster = merged_with_cluster.groupby('cluster')['event_name'].value_counts().groupby(level=0).idxmax()
popular_by_cluster = popular_by_cluster.apply(lambda x: x[1]).reset_index()
popular_by_cluster.columns = ['cluster', 'top_event']
popular_by_cluster.to_csv('popular_by_cluster.csv', index=False)
print('Popular events by cluster saved to popular_by_cluster.csv')

# 4. Default Category Trends by Time of Day

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

merged_df['hour'] = merged_df['eventtimestamp'].dt.hour
merged_df['time_of_day'] = merged_df['hour'].apply(get_time_of_day)

popular_by_time = merged_df.groupby('time_of_day')['event_name'].value_counts().groupby(level=0).idxmax()
popular_by_time = popular_by_time.apply(lambda x: x[1]).reset_index()
popular_by_time.columns = ['time_of_day', 'top_event']
popular_by_time.to_csv('popular_by_time.csv', index=False)
print('Popular events by time of day saved to popular_by_time.csv')

# 5. Default Category Trends by Device Type

popular_by_device = merged_df.groupby('device')['event_name'].value_counts().groupby(level=0).idxmax()
popular_by_device = popular_by_device.apply(lambda x: x[1]).reset_index()
popular_by_device.columns = ['device', 'top_event']
popular_by_device.to_csv('popular_by_device.csv', index=False)
print('Popular events by device saved to popular_by_device.csv')

# Save preprocessed data
merged_df.to_csv('preprocessed_sessions.csv', index=False)

print('Preprocessing complete. Output saved to preprocessed_sessions.csv')

def safe_str(val):
    if pd.isna(val):
        return ''
    if isinstance(val, pd.Timestamp) or isinstance(val, pd.Timedelta):
        return str(val)
    return str(val)

def safe_int(val):
    try:
        if pd.isna(val):
            return None
        return int(val)
    except Exception:
        return None

def safe_float(val):
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None

def safe_bool(val):
    if pd.isna(val):
        return None
    if isinstance(val, str):
        return val.lower() in ['true', '1', 'yes']
    return bool(val)

def migrate_csv_to_db(csv_path='preprocessed_sessions.csv'):
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    db = SessionLocal()
    for _, row in df.iterrows():
        # Convert time_diff to float seconds if possible
        time_diff_val = row.get('time_diff', 0)
        time_diff_float = None
        if isinstance(time_diff_val, str) and 'days' in time_diff_val:
            try:
                t = pd.to_timedelta(time_diff_val)
                time_diff_float = t.total_seconds()
            except Exception:
                time_diff_float = None
        elif time_diff_val is not None and not (isinstance(time_diff_val, float) and pd.isna(time_diff_val)):
            try:
                time_diff_float = float(time_diff_val)
            except Exception:
                time_diff_float = None
        # If time_diff_val is None or NaN, time_diff_float remains None
        session = SessionDB(
            user_pseudo_id=safe_str(row.get('user_pseudo_id', '')),
            session_id=safe_int(row.get('session_id', 0)),
            eventtimestamp=safe_str(row.get('eventtimestamp', '')),
            event_name=safe_str(row.get('event_name', '')),
            transaction_id=safe_str(row.get('transaction_id', '')),
            prev_event_time=safe_str(row.get('prev_event_time', '')),
            time_diff=time_diff_float,
            new_session=safe_bool(row.get('new_session', False)),
            engagement_type=safe_str(row.get('engagement_type', '')),
        )
        db.add(session)
    db.commit()
    db.close()
    print(f"Migrated {len(df)} rows from {csv_path} to SQLite database.")

if __name__ == "__main__":
    migrate_csv_to_db()

# --- Personalization Logic Development ---

# Load cold start outputs
popular_by_cluster = pd.read_csv('popular_by_cluster.csv')
popular_by_demo = pd.read_csv('popular_by_demo.csv')

def get_cta(engagement):
    if engagement == 'high':
        return 'Buy Now'
    elif engagement == 'medium':
        return 'Explore'
    else:
        return 'Discover'

personalization = []
for _, row in user_segments_df.iterrows():
    # Hero banner: top event for user's cluster
    cluster = row.get('cluster', None)
    hero_banner = 'default_banner'
    try:
        cluster_mask_raw = (popular_by_cluster['cluster'].astype(str) == str(cluster))
        if isinstance(cluster_mask_raw, pd.Series):
            cluster_mask = cluster_mask_raw.to_numpy()
            if cluster_mask.dtype == bool and np.any(cluster_mask):
                hero_banner = popular_by_cluster.loc[cluster_mask, 'top_event'].iloc[0]
        elif isinstance(cluster_mask_raw, np.ndarray):
            if cluster_mask_raw.dtype == bool and np.any(cluster_mask_raw):
                hero_banner = popular_by_cluster.loc[cluster_mask_raw, 'top_event'].iloc[0]
        elif isinstance(cluster_mask_raw, bool):
            if cluster_mask_raw:
                hero_banner = popular_by_cluster.loc[cluster_mask_raw, 'top_event'].iloc[0]
    except Exception:
        pass  # use default if mask conversion fails
    # Product module: top event for user's age/gender
    age = row.get('age', None)
    gender = row.get('gender', None)
    product_module = 'default_product'
    try:
        demo_mask_raw = (popular_by_demo['age'].astype(str) == str(age)) & (popular_by_demo['gender'].astype(str) == str(gender))
        if isinstance(demo_mask_raw, pd.Series):
            demo_mask = demo_mask_raw.to_numpy()
            if demo_mask.dtype == bool and np.any(demo_mask):
                product_module = popular_by_demo.loc[demo_mask, 'top_event'].iloc[0]
        elif isinstance(demo_mask_raw, np.ndarray):
            if demo_mask_raw.dtype == bool and np.any(demo_mask_raw):
                product_module = popular_by_demo.loc[demo_mask_raw, 'top_event'].iloc[0]
        elif isinstance(demo_mask_raw, bool):
            if demo_mask_raw:
                product_module = popular_by_demo.loc[demo_mask_raw, 'top_event'].iloc[0]
    except Exception:
        pass  # use default if mask conversion fails
    # CTA module: based on engagement
    cta = get_cta(row.get('engagement_type', 'low'))
    personalization.append({
        'user_pseudo_id': row['user_pseudo_id'],
        'hero_banner': hero_banner,
        'product_module': product_module,
        'cta_module': cta
    })

personalization_df = pd.DataFrame(personalization)
personalization_df.to_csv('personalization_recommendations.csv', index=False)
print('Personalization logic complete. Output saved to personalization_recommendations.csv')
