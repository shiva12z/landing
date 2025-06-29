import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class UserSegmenter:
    def __init__(self, sessions_file: str = 'processed_data/user_sessions.csv', 
                 merged_file: str = 'processed_data/merged_activity_transactions.csv'):
        """
        Initialize the UserSegmenter with processed session and merged data.
        
        Args:
            sessions_file (str): Path to the processed sessions CSV file
            merged_file (str): Path to the merged activity-transactions CSV file
        """
        self.sessions_file = sessions_file
        self.merged_file = merged_file
        self.sessions_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.user_segments: Optional[pd.DataFrame] = None
        
    def load_data(self):
        """
        Load the processed session and merged data.
        """
        print("Loading processed data for user segmentation...")
        
        try:
            # Load session data
            self.sessions_df = pd.read_csv(self.sessions_file)
            print(f"Loaded {len(self.sessions_df)} sessions")
            
            # Load merged data for additional attributes
            self.merged_df = pd.read_csv(self.merged_file)
            print(f"Loaded {len(self.merged_df)} merged records")
            
            # Convert date columns
            if 'session_start' in self.sessions_df.columns:
                self.sessions_df['session_start'] = pd.to_datetime(self.sessions_df['session_start'])
            if 'session_end' in self.sessions_df.columns:
                self.sessions_df['session_end'] = pd.to_datetime(self.sessions_df['session_end'])
            
            return self
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run data_preprocessing.py first to generate the required files.")
            return None
    
    def segment_by_engagement_level(self) -> pd.DataFrame:
        """
        Segment users by engagement levels: cart abandoners, frequent viewers, repeat purchasers.
        
        Returns:
            pd.DataFrame: User engagement segments
        """
        if self.sessions_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print("\nSegmenting users by engagement level...")
        
        # Calculate user-level engagement metrics
        user_engagement = self.sessions_df.groupby('user_pseudo_id').agg({
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
            # Repeat purchasers (multiple transactions)
            if row['total_transactions'] >= 2:
                return 'Repeat Purchaser'
            
            # Single purchasers
            elif row['total_transactions'] == 1:
                return 'Single Purchaser'
            
            # Cart abandoners (visited cart but no purchase)
            elif any('Cart Session' in str(cat) for cat in row['session_categories']):
                return 'Cart Abandoner'
            
            # High engagement viewers (long sessions, many events, no purchase)
            elif (row['avg_session_duration'] >= 10 or 
                  row['avg_events_per_session'] >= 8 or
                  any('High Engagement Session' in str(cat) for cat in row['session_categories'])):
                return 'High Engagement Viewer'
            
            # Frequent viewers (multiple sessions, moderate engagement)
            elif len(row['session_categories']) >= 3:
                return 'Frequent Viewer'
            
            # Occasional viewers (few sessions, low engagement)
            else:
                return 'Occasional Viewer'
        
        user_engagement['engagement_segment'] = user_engagement.apply(categorize_engagement, axis=1)
        
        # Add engagement score (0-100)
        def calculate_engagement_score(row):
            score = 0
            
            # Transaction weight (40 points)
            if row['total_transactions'] >= 2:
                score += 40
            elif row['total_transactions'] == 1:
                score += 20
            
            # Session duration weight (25 points)
            if row['avg_session_duration'] >= 15:
                score += 25
            elif row['avg_session_duration'] >= 10:
                score += 20
            elif row['avg_session_duration'] >= 5:
                score += 15
            elif row['avg_session_duration'] >= 2:
                score += 10
            
            # Event count weight (20 points)
            if row['avg_events_per_session'] >= 10:
                score += 20
            elif row['avg_events_per_session'] >= 8:
                score += 15
            elif row['avg_events_per_session'] >= 5:
                score += 10
            elif row['avg_events_per_session'] >= 3:
                score += 5
            
            # Session frequency weight (15 points)
            session_count = len(row['session_categories'])
            if session_count >= 5:
                score += 15
            elif session_count >= 3:
                score += 10
            elif session_count >= 2:
                score += 5
            
            return min(score, 100)
        
        user_engagement['engagement_score'] = user_engagement.apply(calculate_engagement_score, axis=1)
        
        return user_engagement
    
    def segment_by_demographics(self) -> pd.DataFrame:
        """
        Segment users by demographic attributes: age groups, gender, income brackets.
        
        Returns:
            pd.DataFrame: User demographic segments
        """
        if self.merged_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print("Segmenting users by demographics...")
        
        # Get unique user demographics
        user_demographics = self.merged_df.groupby('user_pseudo_id').agg({
            'Age': 'first',
            'gender': 'first',
            'income_group': 'first'
        }).reset_index()
        
        # Clean and categorize age groups
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
        
        # Clean gender
        user_demographics['gender_clean'] = user_demographics['gender'].fillna('Unknown').str.lower()
        
        # Clean income brackets
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
        
        return user_demographics
    
    def segment_by_traffic_source(self) -> pd.DataFrame:
        """
        Segment users by traffic source: paid vs organic traffic, social channels.
        
        Returns:
            pd.DataFrame: User traffic source segments
        """
        if self.merged_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print("Segmenting users by traffic source...")
        
        # Get unique user traffic sources
        user_sources = self.merged_df.groupby('user_pseudo_id').agg({
            'source': lambda x: list(x.unique()),
            'medium': lambda x: list(x.unique())
        }).reset_index()
        
        # Categorize traffic sources
        def categorize_traffic_source(row):
            sources = [str(s).lower() for s in row['source']]
            mediums = [str(m).lower() for m in row['medium']]
            
            # Social media channels
            social_channels = ['facebook', 'instagram', 'twitter', 'linkedin', 'tiktok', 'youtube']
            if any(channel in sources for channel in social_channels):
                return 'Social Media'
            
            # Paid traffic
            paid_indicators = ['paid', 'cpc', 'cpm', 'paidsearch', 'paidsocial']
            if any(indicator in mediums for indicator in paid_indicators):
                return 'Paid Traffic'
            
            # Organic search
            if 'google' in sources or 'organic' in mediums:
                return 'Organic Search'
            
            # Direct traffic
            if 'direct' in sources or 'none' in mediums:
                return 'Direct Traffic'
            
            # Email marketing
            if 'email' in sources or 'email' in mediums:
                return 'Email Marketing'
            
            # Referral traffic
            if 'referral' in mediums:
                return 'Referral Traffic'
            
            return 'Other'
        
        user_sources['traffic_source_category'] = user_sources.apply(categorize_traffic_source, axis=1)
        
        # Add primary source
        def get_primary_source(row):
            sources = [str(s).lower() for s in row['source']]
            if 'facebook' in sources:
                return 'Facebook'
            elif 'google' in sources:
                return 'Google'
            elif 'instagram' in sources:
                return 'Instagram'
            elif 'direct' in sources:
                return 'Direct'
            elif 'email' in sources:
                return 'Email'
            else:
                return 'Other'
        
        user_sources['primary_source'] = user_sources.apply(get_primary_source, axis=1)
        
        return user_sources
    
    def segment_by_geography(self) -> pd.DataFrame:
        """
        Segment users by geographic attributes: city, state, country.
        
        Returns:
            pd.DataFrame: User geographic segments
        """
        if self.merged_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print("Segmenting users by geography...")
        
        # Get unique user geographic data
        user_geography = self.merged_df.groupby('user_pseudo_id').agg({
            'city': 'first',
            'region': 'first',
            'country': 'first'
        }).reset_index()
        
        # Clean geographic data
        user_geography['city_clean'] = user_geography['city'].fillna('Unknown')
        user_geography['state_clean'] = user_geography['region'].fillna('Unknown')
        user_geography['country_clean'] = user_geography['country'].fillna('Unknown')
        
        # Categorize by region (for US states)
        def categorize_us_region(state):
            if pd.isna(state):
                return 'Unknown'
            
            state = str(state).lower()
            northeast = ['connecticut', 'maine', 'massachusetts', 'new hampshire', 'rhode island', 
                        'vermont', 'new jersey', 'new york', 'pennsylvania']
            midwest = ['illinois', 'indiana', 'michigan', 'ohio', 'wisconsin', 'iowa', 'kansas', 
                      'minnesota', 'missouri', 'nebraska', 'north dakota', 'south dakota']
            south = ['delaware', 'florida', 'georgia', 'maryland', 'north carolina', 'south carolina', 
                    'virginia', 'west virginia', 'alabama', 'kentucky', 'mississippi', 'tennessee', 
                    'arkansas', 'louisiana', 'oklahoma', 'texas']
            west = ['arizona', 'colorado', 'idaho', 'montana', 'nevada', 'new mexico', 'utah', 
                   'wyoming', 'alaska', 'california', 'hawaii', 'oregon', 'washington']
            
            if state in northeast:
                return 'Northeast'
            elif state in midwest:
                return 'Midwest'
            elif state in south:
                return 'South'
            elif state in west:
                return 'West'
            else:
                return 'Other'
        
        user_geography['us_region'] = user_geography['state_clean'].apply(categorize_us_region)
        
        return user_geography
    
    def create_comprehensive_segments(self) -> pd.DataFrame:
        """
        Create comprehensive user segments combining all segmentation dimensions.
        
        Returns:
            pd.DataFrame: Comprehensive user segments
        """
        print("\nCreating comprehensive user segments...")
        
        # Get all segmentation data
        engagement_segments = self.segment_by_engagement_level()
        demographic_segments = self.segment_by_demographics()
        traffic_segments = self.segment_by_traffic_source()
        geography_segments = self.segment_by_geography()
        
        # Merge all segments
        comprehensive_segments = engagement_segments.merge(
            demographic_segments, on='user_pseudo_id', how='left'
        ).merge(
            traffic_segments, on='user_pseudo_id', how='left'
        ).merge(
            geography_segments, on='user_pseudo_id', how='left'
        )
        
        # Create high-level customer segments
        def create_customer_segment(row):
            # High-value customers
            if (row['total_revenue'] >= 500 and row['total_transactions'] >= 2):
                return 'High-Value Customer'
            
            # Loyal customers
            elif (row['engagement_score'] >= 70 and row['total_transactions'] >= 1):
                return 'Loyal Customer'
            
            # Potential customers (high engagement, no purchase)
            elif (row['engagement_score'] >= 60 and row['total_transactions'] == 0):
                return 'High-Potential Customer'
            
            # At-risk customers (previous purchases, low recent engagement)
            elif (row['total_transactions'] >= 1 and row['engagement_score'] <= 30):
                return 'At-Risk Customer'
            
            # New customers
            elif (row['total_transactions'] == 1 and row['engagement_score'] <= 50):
                return 'New Customer'
            
            # Casual browsers
            elif (row['engagement_score'] <= 40 and row['total_transactions'] == 0):
                return 'Casual Browser'
            
            else:
                return 'Regular Customer'
        
        comprehensive_segments['customer_segment'] = comprehensive_segments.apply(create_customer_segment, axis=1)
        
        # Add segment priority for marketing
        segment_priority = {
            'High-Value Customer': 1,
            'Loyal Customer': 2,
            'High-Potential Customer': 3,
            'At-Risk Customer': 4,
            'New Customer': 5,
            'Regular Customer': 6,
            'Casual Browser': 7
        }
        
        comprehensive_segments['segment_priority'] = comprehensive_segments['customer_segment'].map(lambda x: segment_priority.get(x, 8))
        
        self.user_segments = comprehensive_segments
        
        return comprehensive_segments
    
    def generate_segment_insights(self) -> Dict:
        """
        Generate insights and statistics for each segment.
        
        Returns:
            Dict: Segment insights and statistics
        """
        if self.user_segments is None:
            raise ValueError("User segments not created. Please call create_comprehensive_segments() first.")
        
        print("\nGenerating segment insights...")
        
        insights = {}
        
        # Engagement segment insights
        engagement_stats = self.user_segments['engagement_segment'].value_counts()
        insights['engagement_segments'] = {
            'distribution': engagement_stats.to_dict(),
            'total_users': len(self.user_segments),
            'top_segment': engagement_stats.index[0],
            'top_segment_percentage': (engagement_stats.iloc[0] / len(self.user_segments)) * 100
        }
        
        # Customer segment insights
        customer_stats = self.user_segments['customer_segment'].value_counts()
        insights['customer_segments'] = {
            'distribution': customer_stats.to_dict(),
            'high_value_count': len(self.user_segments[self.user_segments['customer_segment'] == 'High-Value Customer']),
            'loyal_count': len(self.user_segments[self.user_segments['customer_segment'] == 'Loyal Customer']),
            'potential_count': len(self.user_segments[self.user_segments['customer_segment'] == 'High-Potential Customer'])
        }
        
        # Demographic insights
        age_stats = self.user_segments['age_group'].value_counts()
        gender_stats = self.user_segments['gender_clean'].value_counts()
        income_stats = self.user_segments['income_bracket'].value_counts()
        
        insights['demographics'] = {
            'age_distribution': age_stats.to_dict(),
            'gender_distribution': gender_stats.to_dict(),
            'income_distribution': income_stats.to_dict(),
            'dominant_age_group': age_stats.index[0],
            'dominant_gender': gender_stats.index[0],
            'dominant_income_bracket': income_stats.index[0]
        }
        
        # Traffic source insights
        traffic_stats = self.user_segments['traffic_source_category'].value_counts()
        source_stats = self.user_segments['primary_source'].value_counts()
        
        insights['traffic_sources'] = {
            'traffic_category_distribution': traffic_stats.to_dict(),
            'primary_source_distribution': source_stats.to_dict(),
            'top_traffic_source': traffic_stats.index[0],
            'top_primary_source': source_stats.index[0]
        }
        
        # Geographic insights
        country_stats = self.user_segments['country_clean'].value_counts()
        region_stats = self.user_segments['us_region'].value_counts()
        
        insights['geography'] = {
            'country_distribution': country_stats.to_dict(),
            'us_region_distribution': region_stats.to_dict(),
            'top_country': country_stats.index[0],
            'top_us_region': region_stats.index[0] if region_stats.index[0] != 'Other' else region_stats.index[1]
        }
        
        # Revenue insights by segment (simplified to avoid idxmax issues)
        revenue_by_segment = self.user_segments.groupby('customer_segment')['total_revenue'].sum()
        insights['revenue_analysis'] = {
            'revenue_by_segment': revenue_by_segment.to_dict(),
            'total_revenue': self.user_segments['total_revenue'].sum(),
            'average_revenue_per_user': self.user_segments['total_revenue'].mean(),
            'top_revenue_segment': revenue_by_segment.index[0] if len(revenue_by_segment) > 0 else 'Unknown'
        }
        
        return insights
    
    def save_segments(self, output_dir: str = 'user_segments'):
        """
        Save user segments and insights to files.
        
        Args:
            output_dir (str): Directory to save segment data
        """
        import os
        
        if self.user_segments is None:
            raise ValueError("User segments not created. Please call create_comprehensive_segments() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving user segments to {output_dir}...")
        
        # Save comprehensive segments
        self.user_segments.to_csv(f'{output_dir}/comprehensive_user_segments.csv', index=False)
        print(f"Saved comprehensive segments: {output_dir}/comprehensive_user_segments.csv")
        
        # Save segment insights
        insights = self.generate_segment_insights()
        import json
        with open(f'{output_dir}/segment_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        print(f"Saved segment insights: {output_dir}/segment_insights.json")
        
        # Save segment summaries
        self._save_segment_summaries(output_dir)
        
        return self
    
    def _save_segment_summaries(self, output_dir: str):
        """
        Save detailed summaries for each segment.
        
        Args:
            output_dir (str): Directory to save summaries
        """
        if self.user_segments is None:
            raise ValueError("User segments not created. Please call create_comprehensive_segments() first.")
        
        # Engagement segment summary
        engagement_summary = self.user_segments.groupby('engagement_segment').agg({
            'user_pseudo_id': 'count',
            'total_revenue': ['sum', 'mean'],
            'engagement_score': 'mean',
            'total_transactions': 'sum'
        }).round(2)
        
        engagement_summary.to_csv(f'{output_dir}/engagement_segment_summary.csv')
        
        # Customer segment summary
        customer_summary = self.user_segments.groupby('customer_segment').agg({
            'user_pseudo_id': 'count',
            'total_revenue': ['sum', 'mean'],
            'engagement_score': 'mean',
            'total_transactions': 'sum'
        }).round(2)
        
        customer_summary.to_csv(f'{output_dir}/customer_segment_summary.csv')
        
        print(f"Saved segment summaries to {output_dir}/")
    
    def run_full_segmentation(self, save_data: bool = True):
        """
        Run the complete user segmentation pipeline.
        
        Args:
            save_data (bool): Whether to save segment data to files
        """
        print("Starting User Segmentation Pipeline")
        print("=" * 50)
        
        try:
            # Load data
            if self.load_data() is None:
                return None
            
            # Create comprehensive segments
            self.create_comprehensive_segments()
            
            # Generate insights
            insights = self.generate_segment_insights()
            
            # Print key insights
            self._print_segment_insights(insights)
            
            # Save data if requested
            if save_data:
                self.save_segments()
            
            print("\nUser Segmentation Pipeline Completed Successfully!")
            print("=" * 50)
            
            return insights
            
        except Exception as e:
            print(f"Error in segmentation pipeline: {str(e)}")
            raise
    
    def _print_segment_insights(self, insights: Dict):
        """
        Print key segment insights to console.
        
        Args:
            insights (Dict): Segment insights dictionary
        """
        print("\n" + "="*60)
        print("USER SEGMENTATION INSIGHTS")
        print("="*60)
        
        # Customer segments
        print(f"\n1. CUSTOMER SEGMENTS:")
        customer_dist = insights['customer_segments']['distribution']
        for segment, count in customer_dist.items():
            percentage = (count / insights['customer_segments']['total_users']) * 100
            print(f"   {segment}: {count:,} users ({percentage:.1f}%)")
        
        # Engagement segments
        print(f"\n2. ENGAGEMENT SEGMENTS:")
        engagement_dist = insights['engagement_segments']['distribution']
        for segment, count in engagement_dist.items():
            percentage = (count / insights['engagement_segments']['total_users']) * 100
            print(f"   {segment}: {count:,} users ({percentage:.1f}%)")
        
        # Top demographics
        print(f"\n3. DEMOGRAPHICS:")
        print(f"   Dominant Age Group: {insights['demographics']['dominant_age_group']}")
        print(f"   Dominant Gender: {insights['demographics']['dominant_gender']}")
        print(f"   Dominant Income: {insights['demographics']['dominant_income_bracket']}")
        
        # Traffic sources
        print(f"\n4. TRAFFIC SOURCES:")
        print(f"   Top Traffic Category: {insights['traffic_sources']['top_traffic_source']}")
        print(f"   Top Primary Source: {insights['traffic_sources']['top_primary_source']}")
        
        # Revenue insights
        print(f"\n5. REVENUE ANALYSIS:")
        print(f"   Total Revenue: ${insights['revenue_analysis']['total_revenue']:,.2f}")
        print(f"   Average Revenue per User: ${insights['revenue_analysis']['average_revenue_per_user']:.2f}")
        print(f"   Top Revenue Segment: {insights['revenue_analysis']['top_revenue_segment']}")

# Example usage
if __name__ == "__main__":
    # Initialize segmenter
    segmenter = UserSegmenter()
    
    # Run the full segmentation pipeline
    insights = segmenter.run_full_segmentation(save_data=True)
    
    if insights:
        print(f"\nSegmentation completed successfully!")
        print(f"Total users segmented: {insights['engagement_segments']['total_users']:,}") 