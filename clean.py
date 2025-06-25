# Import required libraries
import pandas as pd

# 1. Load the datasets
# Replace with your actual file paths
df_activity = pd.read_csv('user_activity_logs.csv')
df_transactions = pd.read_csv('transactions.csv')

# 2. Check column names to diagnose KeyError
print("Activity Logs Columns:", df_activity.columns.tolist())
print("Transactions Columns:", df_transactions.columns.tolist())

# 3. Fix transaction_id column names if needed
# Example: Rename if case mismatch found (adjust based on output from step 2)
if 'Transaction_id' in df_transactions.columns:
    df_transactions.rename(columns={'Transaction_id': 'transaction_id'}, inplace=True)
if 'Transaction_id' in df_activity.columns:
    df_activity.rename(columns={'Transaction_id': 'transaction_id'}, inplace=True)

# Verify transaction_id exists
if 'transaction_id' not in df_activity.columns or 'transaction_id' not in df_transactions.columns:
    raise ValueError("Column 'transaction_id' not found in one or both datasets. Check dataset files.")

# 4. Ensure consistent data types for transaction_id
df_activity['transaction_id'] = df_activity['transaction_id'].astype(str)
df_transactions['transaction_id'] = df_transactions['transaction_id'].astype(str)

# 5. Explore the datasets
print("\nActivity Logs Info:")
print(df_activity.info())
print("\nActivity Logs Sample:")
print(df_activity.head(10))
print("\nTransactions Info:")
print(df_transactions.info())
print("\nTransactions Sample:")
print(df_transactions.head(10))
print("\nDataset Sizes:")
print(f"Activity Logs: {df_activity.shape}, Transactions: {df_transactions.shape}")

# 6. Understand schema (document key columns)
schema_notes = {
    'user_pseudo_id': 'Unique user identifier for session grouping',
    'transaction_id': 'Links activity to transaction dataset',
    'event_name': 'Type of user action (e.g., view_item, purchase)',
    'ItemCategory': 'High-level product classification (e.g., Accessories, Footwear)',
    'eventTimestamp': 'Timestamp of user event'
}
print("\nSchema Notes:")
for key, value in schema_notes.items():
    print(f"{key}: {value}")

# 7. Assess data quality
# Missing values
print("\nMissing Values in Activity Logs:")
print(df_activity.isna().sum())
print("\nMissing Values in Transactions:")
print(df_transactions.isna().sum())

# Handle missing values
df_activity['gender'] = df_activity['gender'].fillna('unknown')
df_activity['income_group'] = df_activity['income_group'].fillna('unknown')
df_transactions['ItemCategory'] = df_transactions['ItemCategory'].fillna('unknown')

# Convert timestamps
df_activity['eventTimestamp'] = pd.to_datetime(df_activity['eventTimestamp'])
df_transactions['Date'] = pd.to_datetime(df_transactions['Date'])

# Check duplicates
print("\nDuplicate Rows in Activity Logs:", df_activity.duplicated().sum())
print("Duplicate Rows in Transactions:", df_transactions.duplicated().sum())
print("Duplicate Transaction IDs:", df_transactions['transaction_id'].duplicated().sum())

# Check outliers (Item_revenue)
print("\nTransaction Revenue Summary:")
print(df_transactions['Item_revenue'].describe())
df_transactions = df_transactions[df_transactions['Item_revenue'] >= 0]

# Check categorical consistency
print("\nUnique Event Names:", df_activity['event_name'].unique())

# 8. Join datasets
df_merged = pd.merge(df_activity, df_transactions, on='transaction_id', how='left')
print("\nMerged Dataset Shape:", df_merged.shape)
print("Missing Item_revenue in Merged (expected for non-purchases):", df_merged['Item_revenue'].isna().sum())

# 9. Construct user sessions
df_activity = df_activity.sort_values(['user_pseudo_id', 'eventTimestamp'])
sessions = df_activity.groupby('user_pseudo_id').agg({
    'eventTimestamp': ['min', 'max'],
    'event_name': ['count', lambda x: 'purchase' in list(x)]
})
sessions.columns = ['start_time', 'end_time', 'event_count', 'has_purchase']
sessions['duration'] = (sessions['end_time'] - sessions['start_time']).dt.total_seconds() / 60
print("\nUser Sessions Sample:")
print(sessions.head())

# Categorize engagement type
def categorize_engagement(events):
    if 'purchase' in events:
        return 'purchaser'
    elif 'add_to_cart' in events:
        return 'cart_abandoner'
    else:
        return 'browser'
session_events = df_activity.groupby('user_pseudo_id')['event_name'].apply(list)
sessions['engagement_type'] = session_events.apply(categorize_engagement)
print("\nEngagement Type Distribution:")
print(sessions['engagement_type'].value_counts())

# 10. Feature engineering
# Behavioral: Category views
category_views = df_merged.groupby(['user_pseudo_id', 'ItemCategory']).size().unstack(fill_value=0)
print("\nCategory Views Sample:")
print(category_views.head())

# Demographic: One-hot encode
demo_features = pd.get_dummies(df_activity[['user_pseudo_id', 'gender', 'income_group']].drop_duplicates('user_pseudo_id'),
                               columns=['gender', 'income_group'])
print("\nDemographic Features Sample:")
print(demo_features.head())

# Temporal: Extract hour and day
df_activity['event_hour'] = df_activity['eventTimestamp'].dt.hour
df_activity['event_day'] = df_activity['eventTimestamp'].dt.day_name()

# Transaction: Revenue
user_revenue = df_merged.groupby('user_pseudo_id')['Item_revenue'].sum().reset_index(name='total_revenue')
category_revenue = df_merged.groupby(['user_pseudo_id', 'ItemCategory'])['Item_revenue'].sum().unstack(fill_value=0)
print("\nUser Revenue Sample:")
print(user_revenue.head())

# 11. Exploratory Data Analysis (EDA)
event_counts = df_activity['event_name'].value_counts()
print("\nEvent Distribution:")
print(event_counts)

category_revenue_sum = df_transactions.groupby('ItemCategory')['Item_revenue'].sum()
print("\nTotal Revenue by Item Category:")
print(category_revenue_sum)

device_engagement = df_activity.merge(sessions[['engagement_type']], left_on='user_pseudo_id', right_index=True)
device_engagement_counts = device_engagement.groupby(['category', 'engagement_type']).size()
print("\nEngagement by Device Category:")
print(device_engagement_counts)

numerical_cols = ['event_hour', 'total_revenue']
merged_features = df_activity.merge(user_revenue, on='user_pseudo_id')
correlation_matrix = merged_features[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 12. Save cleaned data
df_merged.to_csv('cleaned_merged_data.csv', index=False)
sessions.to_csv('user_sessions.csv')
category_views.to_csv('category_views.csv')
demo_features.to_csv('demo_features.csv')
user_revenue.to_csv('user_revenue.csv')
print("\nCleaned datasets saved: cleaned_merged_data.csv, user_sessions.csv, category_views.csv, demo_features.csv, user_revenue.csv")

# Summary
print("\nKey Findings:")
print(f"- Unique Users: {df_activity['user_pseudo_id'].nunique()}")
print(f"- Top Category by Revenue: {category_revenue_sum.idxmax()}")
print(f"- Most Common Event: {event_counts.idxmax()}")