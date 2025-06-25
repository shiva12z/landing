import pandas as pd
import os
from app.db import SessionLocal
from app.models.session_db import SessionDB

# Load datasets
activity_df = pd.read_csv('dataset1_final.csv')
transaction_df = pd.read_csv('dataset2_final.csv')

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
        if isinstance(time_diff_val, str) and 'days' in time_diff_val:
            try:
                t = pd.to_timedelta(time_diff_val)
                time_diff_float = t.total_seconds()
            except Exception:
                time_diff_float = None
        elif pd.isna(time_diff_val):
            time_diff_float = None
        else:
            try:
                time_diff_float = float(time_diff_val)
            except Exception:
                time_diff_float = None
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
