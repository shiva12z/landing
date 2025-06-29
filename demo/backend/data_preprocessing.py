import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Optional
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, activity_file='dataset1_final.csv', transaction_file='dataset2_final.csv'):
        """
        Initialize the DataPreprocessor with file paths for activity and transaction datasets.
        
        Args:
            activity_file (str): Path to the activity dataset
            transaction_file (str): Path to the transaction dataset
        """
        self.activity_file = activity_file
        self.transaction_file = transaction_file
        self.activity_df: Optional[pd.DataFrame] = None
        self.transaction_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.sessions_df: Optional[pd.DataFrame] = None
        
    def load_data(self):
        """
        Load both activity and transaction datasets.
        """
        print("Loading datasets...")
        
        # Load activity dataset
        print(f"Loading activity data from {self.activity_file}...")
        self.activity_df = pd.read_csv(self.activity_file)
        print(f"Activity dataset shape: {self.activity_df.shape}")
        print(f"Activity dataset columns: {list(self.activity_df.columns)}")
        
        # Load transaction dataset
        print(f"Loading transaction data from {self.transaction_file}...")
        self.transaction_df = pd.read_csv(self.transaction_file)
        print(f"Transaction dataset shape: {self.transaction_df.shape}")
        print(f"Transaction dataset columns: {list(self.transaction_df.columns)}")
        
        return self
    
    def clean_data(self):
        """
        Clean and preprocess the datasets.
        """
        print("\nCleaning datasets...")
        
        # Check if data is loaded
        if self.activity_df is None or self.transaction_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        # Clean activity dataset
        print("Cleaning activity dataset...")
        
        # Convert date columns to datetime
        self.activity_df['eventDate'] = pd.to_datetime(self.activity_df['eventDate'])
        self.activity_df['eventTimestamp'] = pd.to_datetime(self.activity_df['eventTimestamp'])
        
        # Handle missing values
        self.activity_df['transaction_id'] = self.activity_df['transaction_id'].fillna('')
        
        # Remove rows with missing user_pseudo_id
        initial_rows = len(self.activity_df)
        self.activity_df = self.activity_df.dropna(subset=['user_pseudo_id'])
        print(f"Removed {initial_rows - len(self.activity_df)} rows with missing user_pseudo_id")
        
        # Clean transaction dataset
        print("Cleaning transaction dataset...")
        
        # Convert date column to datetime
        self.transaction_df['Date'] = pd.to_datetime(self.transaction_df['Date'])
        
        # Handle missing values
        self.transaction_df = self.transaction_df.dropna(subset=['Transaction_ID'])
        
        print(f"Activity dataset shape after cleaning: {self.activity_df.shape}")
        print(f"Transaction dataset shape after cleaning: {self.transaction_df.shape}")
        
        return self
    
    def join_datasets(self):
        """
        Join activity and transaction datasets using transaction_id field.
        """
        print("\nJoining datasets...")
        
        # Check if data is cleaned
        if self.activity_df is None or self.transaction_df is None:
            raise ValueError("Data not cleaned. Please call clean_data() first.")
        
        # Standardize transaction ID column names for joining
        self.activity_df['transaction_id_clean'] = self.activity_df['transaction_id'].astype(str)
        self.transaction_df['Transaction_ID_clean'] = self.transaction_df['Transaction_ID'].astype(str)
        
        # Perform left join to keep all activity records
        self.merged_df = self.activity_df.merge(
            self.transaction_df,
            left_on='transaction_id_clean',
            right_on='Transaction_ID_clean',
            how='left',
            suffixes=('', '_transaction')
        )
        
        print(f"Merged dataset shape: {self.merged_df.shape}")
        print(f"Number of records with transactions: {self.merged_df['Transaction_ID'].notna().sum()}")
        print(f"Number of records without transactions: {self.merged_df['Transaction_ID'].isna().sum()}")
        
        return self
    
    def construct_sessions(self, session_timeout_minutes=30):
        """
        Construct meaningful user sessions based on time gaps.
        
        Args:
            session_timeout_minutes (int): Minutes of inactivity to consider a new session
        """
        print(f"\nConstructing user sessions (timeout: {session_timeout_minutes} minutes)...")
        
        # Check if data is merged
        if self.merged_df is None:
            raise ValueError("Data not merged. Please call join_datasets() first.")
        
        # Sort by user and timestamp
        self.merged_df = self.merged_df.sort_values(['user_pseudo_id', 'eventTimestamp'])
        
        # Calculate time difference between consecutive events for each user
        self.merged_df['time_diff'] = self.merged_df.groupby('user_pseudo_id')['eventTimestamp'].diff()
        
        # Create session breaks (when time difference > timeout)
        session_break = self.merged_df['time_diff'] > timedelta(minutes=session_timeout_minutes)
        
        # Create session IDs
        self.merged_df['session_id'] = (session_break | 
                                       self.merged_df['user_pseudo_id'] != self.merged_df['user_pseudo_id'].shift(1)).cumsum()
        
        # Create unique session identifiers
        self.merged_df['unique_session_id'] = self.merged_df['user_pseudo_id'].astype(str) + '_' + self.merged_df['session_id'].astype(str)
        
        print(f"Total sessions created: {self.merged_df['unique_session_id'].nunique()}")
        print(f"Average events per session: {len(self.merged_df) / self.merged_df['unique_session_id'].nunique():.2f}")
        
        return self
    
    def categorize_sessions(self):
        """
        Categorize sessions based on engagement type and behavior patterns.
        """
        print("\nCategorizing sessions...")
        
        # Check if sessions are constructed
        if self.merged_df is None:
            raise ValueError("Sessions not constructed. Please call construct_sessions() first.")
        
        # Create session-level aggregations
        session_metrics = self.merged_df.groupby('unique_session_id').agg({
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
        
        # Categorize sessions based on engagement patterns
        def categorize_session(row):
            # Check if session has transactions
            if row['transaction_count'] > 0:
                return 'Purchase Session'
            
            # Check for high engagement (many events, long duration)
            if row['event_count'] >= 10 or row['session_duration_minutes'] >= 15:
                return 'High Engagement Session'
            
            # Check for specific page types
            page_types = row['page_types']
            if any('product' in str(pt).lower() for pt in page_types):
                return 'Product Browsing Session'
            elif any('cart' in str(pt).lower() for pt in page_types):
                return 'Cart Session'
            elif any('checkout' in str(pt).lower() for pt in page_types):
                return 'Checkout Session'
            
            # Check for quick sessions
            if row['event_count'] <= 3 and row['session_duration_minutes'] <= 2:
                return 'Quick Visit Session'
            
            return 'General Browsing Session'
        
        session_metrics['session_category'] = session_metrics.apply(categorize_session, axis=1)
        
        # Add engagement level
        def get_engagement_level(row):
            if row['transaction_count'] > 0:
                return 'High'  # Purchase sessions are always high engagement
            elif row['event_count'] >= 8 or row['session_duration_minutes'] >= 10:
                return 'High'
            elif row['event_count'] >= 4 or row['session_duration_minutes'] >= 5:
                return 'Medium'
            else:
                return 'Low'
        
        session_metrics['engagement_level'] = session_metrics.apply(get_engagement_level, axis=1)
        
        self.sessions_df = session_metrics
        
        # Ensure sessions_df is not None for type checker
        if self.sessions_df is None:
            raise RuntimeError("Failed to create sessions DataFrame")
        
        # Print session category distribution
        print("\nSession Category Distribution:")
        print(self.sessions_df['session_category'].value_counts())
        
        print("\nEngagement Level Distribution:")
        print(self.sessions_df['engagement_level'].value_counts())
        
        return self
    
    def generate_session_summary(self):
        """
        Generate comprehensive summary statistics for sessions.
        """
        print("\nGenerating session summary...")
        
        # Check if sessions are categorized
        if self.sessions_df is None:
            raise ValueError("Sessions not categorized. Please call categorize_sessions() first.")
        
        # Type cast to help the type checker
        sessions: pd.DataFrame = self.sessions_df
        
        summary_stats = {
            'Total Sessions': len(sessions),
            'Total Users': sessions['user_pseudo_id'].nunique(),
            'Sessions with Purchases': len(sessions[sessions['transaction_count'] > 0]),
            'Average Session Duration (minutes)': sessions['session_duration_minutes'].mean(),
            'Average Events per Session': sessions['event_count'].mean(),
            'Total Revenue': sessions['total_revenue'].sum(),
            'Average Revenue per Purchase Session': sessions[sessions['transaction_count'] > 0]['total_revenue'].mean()
        }
        
        print("\nSession Summary Statistics:")
        for key, value in summary_stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        return summary_stats
    
    def save_processed_data(self, output_dir='processed_data'):
        """
        Save processed data to files.
        
        Args:
            output_dir (str): Directory to save processed data
        """
        import os
        
        # Check if data is processed
        if self.merged_df is None or self.sessions_df is None:
            raise ValueError("Data not processed. Please run the full pipeline first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}...")
        
        # Save merged dataset
        self.merged_df.to_csv(f'{output_dir}/merged_activity_transactions.csv', index=False)
        print(f"Saved merged dataset: {output_dir}/merged_activity_transactions.csv")
        
        # Save session-level data
        self.sessions_df.to_csv(f'{output_dir}/user_sessions.csv', index=False)
        print(f"Saved session data: {output_dir}/user_sessions.csv")
        
        return self
    
    def run_full_pipeline(self, session_timeout_minutes=30, save_data=True):
        """
        Run the complete data preprocessing pipeline.
        
        Args:
            session_timeout_minutes (int): Minutes of inactivity to consider a new session
            save_data (bool): Whether to save processed data to files
        """
        print("Starting Data Preprocessing Pipeline")
        print("=" * 50)
        
        try:
            (self.load_data()
                 .clean_data()
                 .join_datasets()
                 .construct_sessions(session_timeout_minutes)
                 .categorize_sessions()
                 .generate_session_summary())
            
            if save_data:
                self.save_processed_data()
            
            print("\nData Preprocessing Pipeline Completed Successfully!")
            print("=" * 50)
            
        except Exception as e:
            print(f"Error in preprocessing pipeline: {str(e)}")
            raise
        
        return self

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run the full pipeline
    preprocessor.run_full_pipeline(session_timeout_minutes=30)
    
    # Access processed data
    print(f"\nProcessed data shapes:")
    if preprocessor.merged_df is not None:
        print(f"Merged dataset: {preprocessor.merged_df.shape}")
    if preprocessor.sessions_df is not None:
        print(f"Sessions dataset: {preprocessor.sessions_df.shape}") 