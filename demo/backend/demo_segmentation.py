import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def demo_user_segmentation():
    """
    Demo function to show the user segmentation workflow with sample data.
    """
    print("="*60)
    print("USER SEGMENTATION DEMO")
    print("="*60)
    
    try:
        # Load a small sample of the processed data for demo
        print("\n1. LOADING PROCESSED DATA SAMPLE")
        print("-" * 40)
        
        # Load session data sample
        print("Loading session data sample...")
        sessions_sample = pd.read_csv('processed_data/user_sessions.csv', nrows=500)
        print(f"Session sample shape: {sessions_sample.shape}")
        
        # Load merged data sample
        print("Loading merged data sample...")
        merged_sample = pd.read_csv('processed_data/merged_activity_transactions.csv', nrows=1000)
        print(f"Merged data sample shape: {merged_sample.shape}")
        
        # Convert date columns
        if 'session_start' in sessions_sample.columns:
            sessions_sample['session_start'] = pd.to_datetime(sessions_sample['session_start'])
        if 'session_end' in sessions_sample.columns:
            sessions_sample['session_end'] = pd.to_datetime(sessions_sample['session_end'])
        
        print("\n2. ENGAGEMENT LEVEL SEGMENTATION")
        print("-" * 40)
        
        # Calculate user-level engagement metrics
        user_engagement = sessions_sample.groupby('user_pseudo_id').agg({
            'session_duration_minutes': ['mean', 'sum'],
            'event_count': ['mean', 'sum'],
            'transaction_count': 'sum',
            'total_revenue': 'sum',
            'session_category': lambda x: list(x),
            'engagement_level': lambda x: list(x)
        }).reset_index()
        
        # Flatten column names
        user_engagement.columns = [
            'user_pseudo_id', 'avg_session_duration', 'total_session_duration',
            'avg_events_per_session', 'total_events', 'total_transactions',
            'total_revenue', 'session_categories', 'engagement_levels'
        ]
        
        # Define engagement segments
        def categorize_engagement(row):
            if row['total_transactions'] >= 2:
                return 'Repeat Purchaser'
            elif row['total_transactions'] == 1:
                return 'Single Purchaser'
            elif any('Cart Session' in str(cat) for cat in row['session_categories']):
                return 'Cart Abandoner'
            elif (row['avg_session_duration'] >= 10 or 
                  row['avg_events_per_session'] >= 8 or
                  any('High Engagement Session' in str(cat) for cat in row['session_categories'])):
                return 'High Engagement Viewer'
            elif len(row['session_categories']) >= 3:
                return 'Frequent Viewer'
            else:
                return 'Occasional Viewer'
        
        user_engagement['engagement_segment'] = user_engagement.apply(categorize_engagement, axis=1)
        
        # Calculate engagement score
        def calculate_engagement_score(row):
            score = 0
            if row['total_transactions'] >= 2:
                score += 40
            elif row['total_transactions'] == 1:
                score += 20
            
            if row['avg_session_duration'] >= 15:
                score += 25
            elif row['avg_session_duration'] >= 10:
                score += 20
            elif row['avg_session_duration'] >= 5:
                score += 15
            elif row['avg_session_duration'] >= 2:
                score += 10
            
            if row['avg_events_per_session'] >= 10:
                score += 20
            elif row['avg_events_per_session'] >= 8:
                score += 15
            elif row['avg_events_per_session'] >= 5:
                score += 10
            elif row['avg_events_per_session'] >= 3:
                score += 5
            
            session_count = len(row['session_categories'])
            if session_count >= 5:
                score += 15
            elif session_count >= 3:
                score += 10
            elif session_count >= 2:
                score += 5
            
            return min(score, 100)
        
        user_engagement['engagement_score'] = user_engagement.apply(calculate_engagement_score, axis=1)
        
        print("Engagement Segments Created:")
        engagement_counts = user_engagement['engagement_segment'].value_counts()
        for segment, count in engagement_counts.items():
            percentage = (count / len(user_engagement)) * 100
            print(f"  {segment}: {count} users ({percentage:.1f}%)")
        
        print("\n3. DEMOGRAPHIC SEGMENTATION")
        print("-" * 40)
        
        # Get user demographics
        user_demographics = merged_sample.groupby('user_pseudo_id').agg({
            'Age': 'first',
            'gender': 'first',
            'income_group': 'first'
        }).reset_index()
        
        # Categorize age groups
        def categorize_age(age_str):
            if pd.isna(age_str):
                return 'Unknown'
            age_str = str(age_str).lower()
            if '18-24' in age_str or '18' in age_str:
                return '18-24'
            elif '25-34' in age_str or '25' in age_str:
                return '25-34'
            elif '35-44' in age_str or '35' in age_str:
                return '35-44'
            elif '45-54' in age_str or '45' in age_str:
                return '45-54'
            elif '55-64' in age_str or '55' in age_str:
                return '55-64'
            elif '65+' in age_str or 'above 64' in age_str:
                return '65+'
            else:
                return 'Unknown'
        
        user_demographics['age_group'] = user_demographics['Age'].apply(categorize_age)
        user_demographics['gender_clean'] = user_demographics['gender'].fillna('Unknown').str.lower()
        
        # Categorize income
        def categorize_income(income_str):
            if pd.isna(income_str):
                return 'Unknown'
            income_str = str(income_str).lower()
            if 'top 10%' in income_str or 'high' in income_str:
                return 'High Income'
            elif '11-20%' in income_str or 'above average' in income_str:
                return 'Above Average'
            elif '21-50%' in income_str or 'average' in income_str:
                return 'Average'
            elif 'below 50%' in income_str or 'low' in income_str:
                return 'Low Income'
            else:
                return 'Unknown'
        
        user_demographics['income_bracket'] = user_demographics['income_group'].apply(categorize_income)
        
        print("Age Group Distribution:")
        age_counts = user_demographics['age_group'].value_counts()
        for age_group, count in age_counts.items():
            percentage = (count / len(user_demographics)) * 100
            print(f"  {age_group}: {count} users ({percentage:.1f}%)")
        
        print("\n4. TRAFFIC SOURCE SEGMENTATION")
        print("-" * 40)
        
        # Get traffic sources
        user_sources = merged_sample.groupby('user_pseudo_id').agg({
            'source': lambda x: list(x.unique()),
            'medium': lambda x: list(x.unique())
        }).reset_index()
        
        # Categorize traffic sources
        def categorize_traffic_source(row):
            sources = [str(s).lower() for s in row['source']]
            mediums = [str(m).lower() for m in row['medium']]
            
            social_channels = ['facebook', 'instagram', 'twitter', 'linkedin', 'tiktok', 'youtube']
            if any(channel in sources for channel in social_channels):
                return 'Social Media'
            
            paid_indicators = ['paid', 'cpc', 'cpm', 'paidsearch', 'paidsocial']
            if any(indicator in mediums for indicator in paid_indicators):
                return 'Paid Traffic'
            
            if 'google' in sources or 'organic' in mediums:
                return 'Organic Search'
            
            if 'direct' in sources or 'none' in mediums:
                return 'Direct Traffic'
            
            if 'email' in sources or 'email' in mediums:
                return 'Email Marketing'
            
            if 'referral' in mediums:
                return 'Referral Traffic'
            
            return 'Other'
        
        user_sources['traffic_source_category'] = user_sources.apply(categorize_traffic_source, axis=1)
        
        print("Traffic Source Distribution:")
        traffic_counts = user_sources['traffic_source_category'].value_counts()
        for source, count in traffic_counts.items():
            percentage = (count / len(user_sources)) * 100
            print(f"  {source}: {count} users ({percentage:.1f}%)")
        
        print("\n5. GEOGRAPHIC SEGMENTATION")
        print("-" * 40)
        
        # Get geographic data
        user_geography = merged_sample.groupby('user_pseudo_id').agg({
            'city': 'first',
            'region': 'first',
            'country': 'first'
        }).reset_index()
        
        user_geography['city_clean'] = user_geography['city'].fillna('Unknown')
        user_geography['state_clean'] = user_geography['region'].fillna('Unknown')
        user_geography['country_clean'] = user_geography['country'].fillna('Unknown')
        
        print("Country Distribution:")
        country_counts = user_geography['country_clean'].value_counts()
        for country, count in country_counts.items():
            percentage = (count / len(user_geography)) * 100
            print(f"  {country}: {count} users ({percentage:.1f}%)")
        
        print("\n6. COMPREHENSIVE CUSTOMER SEGMENTS")
        print("-" * 40)
        
        # Merge all segments
        comprehensive_segments = user_engagement.merge(
            user_demographics, on='user_pseudo_id', how='left'
        ).merge(
            user_sources, on='user_pseudo_id', how='left'
        ).merge(
            user_geography, on='user_pseudo_id', how='left'
        )
        
        # Create customer segments
        def create_customer_segment(row):
            if (row['total_revenue'] >= 500 and row['total_transactions'] >= 2):
                return 'High-Value Customer'
            elif (row['engagement_score'] >= 70 and row['total_transactions'] >= 1):
                return 'Loyal Customer'
            elif (row['engagement_score'] >= 60 and row['total_transactions'] == 0):
                return 'High-Potential Customer'
            elif (row['total_transactions'] >= 1 and row['engagement_score'] <= 30):
                return 'At-Risk Customer'
            elif (row['total_transactions'] == 1 and row['engagement_score'] <= 50):
                return 'New Customer'
            elif (row['engagement_score'] <= 40 and row['total_transactions'] == 0):
                return 'Casual Browser'
            else:
                return 'Regular Customer'
        
        comprehensive_segments['customer_segment'] = comprehensive_segments.apply(create_customer_segment, axis=1)
        
        print("Customer Segments Created:")
        customer_counts = comprehensive_segments['customer_segment'].value_counts()
        for segment, count in customer_counts.items():
            percentage = (count / len(comprehensive_segments)) * 100
            print(f"  {segment}: {count} users ({percentage:.1f}%)")
        
        print("\n7. SEGMENT INSIGHTS")
        print("-" * 40)
        
        # Revenue analysis
        total_revenue = comprehensive_segments['total_revenue'].sum()
        avg_revenue = comprehensive_segments['total_revenue'].mean()
        
        print(f"Total Revenue: ${total_revenue:,.2f}")
        print(f"Average Revenue per User: ${avg_revenue:.2f}")
        
        # Top segments by revenue
        revenue_by_segment = comprehensive_segments.groupby('customer_segment')['total_revenue'].sum().sort_values(ascending=False)
        print(f"\nTop Revenue Segments:")
        for segment, revenue in revenue_by_segment.head(3).items():
            print(f"  {segment}: ${revenue:,.2f}")
        
        # Engagement score distribution
        print(f"\nEngagement Score Distribution:")
        print(f"  High (70-100): {len(comprehensive_segments[comprehensive_segments['engagement_score'] >= 70])} users")
        print(f"  Medium (40-69): {len(comprehensive_segments[(comprehensive_segments['engagement_score'] >= 40) & (comprehensive_segments['engagement_score'] < 70)])} users")
        print(f"  Low (0-39): {len(comprehensive_segments[comprehensive_segments['engagement_score'] < 40])} users")
        
        print("\n" + "="*60)
        print("USER SEGMENTATION DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return comprehensive_segments
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data_preprocessing.py first to generate the required files.")
        return None
    except Exception as e:
        print(f"Error during segmentation demo: {str(e)}")
        return None

if __name__ == "__main__":
    demo_user_segmentation() 