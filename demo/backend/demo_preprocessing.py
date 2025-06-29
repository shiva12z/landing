import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def demo_data_preprocessing():
    """
    Demo function to show the data preprocessing workflow with sample data.
    """
    print("="*60)
    print("DATA PREPROCESSING DEMO")
    print("="*60)
    
    # Step 1: Load and examine datasets
    print("\n1. LOADING DATASETS")
    print("-" * 30)
    
    try:
        # Load a small sample of the datasets for demo
        print("Loading activity dataset sample...")
        activity_sample = pd.read_csv('dataset1_final.csv', nrows=1000)
        print(f"Activity sample shape: {activity_sample.shape}")
        print(f"Activity columns: {list(activity_sample.columns)}")
        
        print("\nLoading transaction dataset sample...")
        transaction_sample = pd.read_csv('dataset2_final.csv', nrows=1000)
        print(f"Transaction sample shape: {transaction_sample.shape}")
        print(f"Transaction columns: {list(transaction_sample.columns)}")
        
        # Step 2: Data cleaning
        print("\n2. DATA CLEANING")
        print("-" * 30)
        
        # Clean activity data
        activity_sample['eventDate'] = pd.to_datetime(activity_sample['eventDate'])
        activity_sample['eventTimestamp'] = pd.to_datetime(activity_sample['eventTimestamp'])
        activity_sample['transaction_id'] = activity_sample['transaction_id'].fillna('')
        
        # Remove rows with missing user_pseudo_id
        initial_rows = len(activity_sample)
        activity_sample = activity_sample.dropna(subset=['user_pseudo_id'])
        print(f"Removed {initial_rows - len(activity_sample)} rows with missing user_pseudo_id")
        
        # Clean transaction data
        transaction_sample['Date'] = pd.to_datetime(transaction_sample['Date'])
        transaction_sample = transaction_sample.dropna(subset=['Transaction_ID'])
        
        print(f"Activity sample after cleaning: {activity_sample.shape}")
        print(f"Transaction sample after cleaning: {transaction_sample.shape}")
        
        # Step 3: Join datasets
        print("\n3. JOINING DATASETS")
        print("-" * 30)
        
        # Standardize transaction ID column names
        activity_sample['transaction_id_clean'] = activity_sample['transaction_id'].astype(str)
        transaction_sample['Transaction_ID_clean'] = transaction_sample['Transaction_ID'].astype(str)
        
        # Perform left join
        merged_sample = activity_sample.merge(
            transaction_sample,
            left_on='transaction_id_clean',
            right_on='Transaction_ID_clean',
            how='left',
            suffixes=('', '_transaction')
        )
        
        print(f"Merged sample shape: {merged_sample.shape}")
        print(f"Records with transactions: {merged_sample['Transaction_ID'].notna().sum()}")
        print(f"Records without transactions: {merged_sample['Transaction_ID'].isna().sum()}")
        
        # Step 4: Construct sessions
        print("\n4. CONSTRUCTING SESSIONS")
        print("-" * 30)
        
        # Sort by user and timestamp
        merged_sample = merged_sample.sort_values(['user_pseudo_id', 'eventTimestamp'])
        
        # Calculate time difference between consecutive events
        merged_sample['time_diff'] = merged_sample.groupby('user_pseudo_id')['eventTimestamp'].diff()
        
        # Create session breaks (30-minute timeout)
        session_break = merged_sample['time_diff'] > timedelta(minutes=30)
        
        # Create session IDs
        merged_sample['session_id'] = (session_break | 
                                     merged_sample['user_pseudo_id'] != merged_sample['user_pseudo_id'].shift(1)).cumsum()
        
        # Create unique session identifiers
        merged_sample['unique_session_id'] = merged_sample['user_pseudo_id'].astype(str) + '_' + merged_sample['session_id'].astype(str)
        
        print(f"Total sessions created: {merged_sample['unique_session_id'].nunique()}")
        print(f"Average events per session: {len(merged_sample) / merged_sample['unique_session_id'].nunique():.2f}")
        
        # Step 5: Categorize sessions
        print("\n5. CATEGORIZING SESSIONS")
        print("-" * 30)
        
        # Create session-level aggregations
        session_metrics = merged_sample.groupby('unique_session_id').agg({
            'user_pseudo_id': 'first',
            'eventTimestamp': ['min', 'max', 'count'],
            'event_name': lambda x: list(x),
            'page_type': lambda x: list(x),
            'Transaction_ID': lambda x: x.notna().sum(),
            'Item_revenue': 'sum',
            'Item_purchase_quantity': 'sum'
        }).reset_index()
        
        # Flatten column names
        session_metrics.columns = [
            'unique_session_id', 'user_pseudo_id', 'session_start', 'session_end', 
            'event_count', 'event_names', 'page_types', 'transaction_count', 
            'total_revenue', 'total_quantity'
        ]
        
        # Calculate session duration
        session_metrics['session_duration_minutes'] = (
            session_metrics['session_end'] - session_metrics['session_start']
        ).dt.total_seconds() / 60
        
        # Categorize sessions
        def categorize_session(row):
            if row['transaction_count'] > 0:
                return 'Purchase Session'
            elif row['event_count'] >= 10 or row['session_duration_minutes'] >= 15:
                return 'High Engagement Session'
            elif any('product' in str(pt).lower() for pt in row['page_types']):
                return 'Product Browsing Session'
            elif any('cart' in str(pt).lower() for pt in row['page_types']):
                return 'Cart Session'
            elif any('checkout' in str(pt).lower() for pt in row['page_types']):
                return 'Checkout Session'
            elif row['event_count'] <= 3 and row['session_duration_minutes'] <= 2:
                return 'Quick Visit Session'
            else:
                return 'General Browsing Session'
        
        session_metrics['session_category'] = session_metrics.apply(categorize_session, axis=1)
        
        # Add engagement level
        def get_engagement_level(row):
            if row['transaction_count'] > 0:
                return 'High'
            elif row['event_count'] >= 8 or row['session_duration_minutes'] >= 10:
                return 'High'
            elif row['event_count'] >= 4 or row['session_duration_minutes'] >= 5:
                return 'Medium'
            else:
                return 'Low'
        
        session_metrics['engagement_level'] = session_metrics.apply(get_engagement_level, axis=1)
        
        # Step 6: Display results
        print("\n6. SESSION ANALYSIS RESULTS")
        print("-" * 30)
        
        print(f"Total Sessions: {len(session_metrics)}")
        print(f"Unique Users: {session_metrics['user_pseudo_id'].nunique()}")
        print(f"Average Session Duration: {session_metrics['session_duration_minutes'].mean():.2f} minutes")
        print(f"Average Events per Session: {session_metrics['event_count'].mean():.2f}")
        
        print(f"\nSession Categories:")
        category_counts = session_metrics['session_category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(session_metrics)) * 100
            print(f"  {category}: {count} sessions ({percentage:.1f}%)")
        
        print(f"\nEngagement Levels:")
        engagement_counts = session_metrics['engagement_level'].value_counts()
        for level, count in engagement_counts.items():
            percentage = (count / len(session_metrics)) * 100
            print(f"  {level}: {count} sessions ({percentage:.1f}%)")
        
        # Purchase analysis
        purchase_sessions = session_metrics[session_metrics['transaction_count'] > 0]
        if len(purchase_sessions) > 0:
            print(f"\nPurchase Analysis:")
            print(f"  Purchase Sessions: {len(purchase_sessions)} ({len(purchase_sessions)/len(session_metrics)*100:.1f}%)")
            print(f"  Total Revenue: ${purchase_sessions['total_revenue'].sum():.2f}")
            print(f"  Average Revenue per Purchase: ${purchase_sessions['total_revenue'].mean():.2f}")
        
        # Step 7: Sample session details
        print(f"\n7. SAMPLE SESSION DETAILS")
        print("-" * 30)
        
        print("Sample sessions from different categories:")
        for category in session_metrics['session_category'].unique():
            sample_session = session_metrics[session_metrics['session_category'] == category].iloc[0]
            print(f"\n{category}:")
            print(f"  User: {sample_session['user_pseudo_id']}")
            print(f"  Duration: {sample_session['session_duration_minutes']:.2f} minutes")
            print(f"  Events: {sample_session['event_count']}")
            print(f"  Engagement: {sample_session['engagement_level']}")
            if sample_session['transaction_count'] > 0:
                print(f"  Revenue: ${sample_session['total_revenue']:.2f}")
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return session_metrics
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset files are in the current directory.")
        return None
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        return None

if __name__ == "__main__":
    demo_data_preprocessing() 