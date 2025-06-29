#!/usr/bin/env python3
"""
Simple Cold Start Strategy Demo
==============================

This demo shows how to use the ColdStartStrategy class to generate
recommendations for new users with minimal setup and complexity.
"""

import pandas as pd
import numpy as np
from cold_start_strategy import ColdStartStrategy
import json

def create_sample_data():
    """
    Create sample data for demonstration if real data is not available.
    """
    print("Creating sample data for demo...")
    
    # Sample user segments data
    np.random.seed(42)
    n_users = 1000
    
    user_segments_data = {
        'user_pseudo_id': [f'user_{i}' for i in range(n_users)],
        'avg_session_duration': np.random.normal(8.5, 3, n_users),
        'avg_events_per_session': np.random.poisson(6, n_users),
        'total_transactions': np.random.poisson(2, n_users),
        'total_revenue': np.random.exponential(100, n_users),
        'engagement_score': np.random.uniform(20, 80, n_users),
        'engagement_segment': np.random.choice(['Low', 'Medium', 'High'], n_users),
        'customer_segment': np.random.choice(['New', 'Returning', 'VIP'], n_users),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_users),
        'gender_clean': np.random.choice(['male', 'female'], n_users),
        'traffic_source_category': np.random.choice(['Social Media', 'Paid Traffic', 'Organic', 'Direct'], n_users)
    }
    
    # Sample sessions data
    n_sessions = 5000
    sessions_data = {
        'user_pseudo_id': np.random.choice(user_segments_data['user_pseudo_id'], n_sessions),
        'session_start': pd.date_range('2024-01-01', periods=n_sessions, freq='H'),
        'session_end': pd.date_range('2024-01-01', periods=n_sessions, freq='H') + pd.Timedelta(hours=1),
        'session_category': np.random.choice(['Clothing', 'Shoes', 'Bags', 'Accessories', 'Jewelry'], n_sessions)
    }
    
    # Sample merged data
    n_merged = 8000
    merged_data = {
        'user_pseudo_id': np.random.choice(user_segments_data['user_pseudo_id'], n_merged),
        'category': np.random.choice(['Clothing', 'Shoes', 'Bags', 'Accessories', 'Jewelry'], n_merged),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_merged),
        'region': np.random.choice(['California', 'Texas', 'New York', 'Florida', 'Illinois'], n_merged),
        'country': ['USA'] * n_merged,
        'source': np.random.choice(['google', 'facebook', 'instagram', 'direct', 'email'], n_merged),
        'medium': np.random.choice(['cpc', 'organic', 'social', 'email', 'direct'], n_merged)
    }
    
    # Create DataFrames
    user_segments_df = pd.DataFrame(user_segments_data)
    sessions_df = pd.DataFrame(sessions_data)
    merged_df = pd.DataFrame(merged_data)
    
    # Save sample data
    import os
    os.makedirs('sample_data', exist_ok=True)
    
    user_segments_df.to_csv('sample_data/user_segments.csv', index=False)
    sessions_df.to_csv('sample_data/user_sessions.csv', index=False)
    merged_df.to_csv('sample_data/merged_activity_transactions.csv', index=False)
    
    print(f"Created sample data:")
    print(f"  - {len(user_segments_df)} user segments")
    print(f"  - {len(sessions_df)} sessions")
    print(f"  - {len(merged_df)} merged records")
    
    return user_segments_df, sessions_df, merged_df

def demo_basic_usage():
    """
    Demonstrate basic usage of the ColdStartStrategy.
    """
    print("\n" + "="*60)
    print("BASIC COLD START STRATEGY DEMO")
    print("="*60)
    
    # Initialize the strategy
    print("\n1. Initializing ColdStartStrategy...")
    cold_start = ColdStartStrategy(
        user_segments_file='sample_data/user_segments.csv',
        sessions_file='sample_data/user_sessions.csv',
        merged_file='sample_data/merged_activity_transactions.csv'
    )
    
    # Load data
    print("\n2. Loading data...")
    result = cold_start.load_data()
    if result is None:
        print("❌ Failed to load data")
        return
    
    print("✅ Data loaded successfully")
    
    # Create user clusters
    print("\n3. Creating user clusters...")
    clusters = cold_start.create_user_clusters(n_clusters=5)
    print(f"✅ Created {len(clusters)} user clusters")
    
    # Build KNN model
    print("\n4. Building KNN model...")
    knn_model = cold_start.build_knn_model(n_neighbors=3)
    print("✅ KNN model built successfully")
    
    # Generate recommendations for a new user
    print("\n5. Generating recommendations for a new user...")
    
    new_user_profile = {
        'avg_session_duration': 10.5,
        'avg_events_per_session': 8,
        'total_transactions': 0,
        'total_revenue': 0,
        'engagement_score': 65,
        'age_group': '25-34',
        'gender_clean': 'female',
        'traffic_source_category': 'Social Media',
        'region': 'California',
        'hour_of_day': 14
    }
    
    recommendations = cold_start.get_cold_start_recommendations(new_user_profile)
    
    # Display results
    print("\n" + "="*60)
    print("RECOMMENDATION RESULTS")
    print("="*60)
    
    final_recs = recommendations['final_recommendations']
    print(f"\n📊 Final Recommendations:")
    print(f"   • Recommended Categories: {final_recs['recommended_categories']}")
    print(f"   • Engagement Level: {final_recs['recommended_engagement_level']}")
    print(f"   • Content Type: {final_recs['recommended_content_type']}")
    print(f"   • Confidence Score: {final_recs['confidence_score']:.2f}")
    print(f"   • Fallback Strategy: {final_recs['fallback_strategy']}")
    
    print(f"\n👥 Similar Users Found: {len(recommendations['similar_users'])}")
    print(f"📈 Demographic Matches: {len(recommendations['demographic_matches'])}")
    
    return cold_start, recommendations

def demo_advanced_features():
    """
    Demonstrate advanced features of the ColdStartStrategy.
    """
    print("\n" + "="*60)
    print("ADVANCED FEATURES DEMO")
    print("="*60)
    
    # Initialize with existing data
    cold_start = ColdStartStrategy(
        user_segments_file='sample_data/user_segments.csv',
        sessions_file='sample_data/user_sessions.csv',
        merged_file='sample_data/merged_activity_transactions.csv'
    )
    
    # Run the full pipeline
    print("\n1. Running full cold start pipeline...")
    results = cold_start.run_full_cold_start_pipeline(save_data=True)
    
    if results:
        print("✅ Full pipeline completed successfully")
        
        # Show cluster information
        print(f"\n2. Cluster Information:")
        clusters = results['user_clusters']
        for cluster_id in clusters.index:
            size = clusters.loc[cluster_id, 'cluster_size']
            engagement = clusters.loc[cluster_id, 'engagement_score']
            print(f"   • Cluster {cluster_id}: {size} users, avg engagement: {engagement:.1f}")
        
        # Show default trends
        print(f"\n3. Default Trends Summary:")
        trends = results['default_trends']
        
        if 'regional' in trends and trends['regional']:
            regions = list(trends['regional'].keys())
            print(f"   • Regional trends available for: {', '.join(regions[:3])}...")
        
        if 'time_of_day' in trends:
            time_periods = list(trends['time_of_day'].keys())
            print(f"   • Time-of-day trends: {', '.join(time_periods)}")
        
        if 'source' in trends:
            sources = list(trends['source'].keys())
            print(f"   • Source trends: {', '.join(sources[:3])}...")
    
    return results

def demo_multiple_users():
    """
    Demonstrate recommendations for multiple different user types.
    """
    print("\n" + "="*60)
    print("MULTIPLE USER TYPES DEMO")
    print("="*60)
    
    # Initialize strategy
    cold_start = ColdStartStrategy(
        user_segments_file='sample_data/user_segments.csv',
        sessions_file='sample_data/user_sessions.csv',
        merged_file='sample_data/merged_activity_transactions.csv'
    )
    
    cold_start.load_data()
    cold_start.create_user_clusters(n_clusters=5)
    cold_start.build_knn_model(n_neighbors=3)
    
    # Different user profiles
    user_profiles = {
        'Young Mobile User': {
            'avg_session_duration': 5.2,
            'avg_events_per_session': 12,
            'total_transactions': 0,
            'total_revenue': 0,
            'engagement_score': 75,
            'age_group': '18-24',
            'gender_clean': 'female',
            'traffic_source_category': 'Social Media',
            'region': 'California',
            'hour_of_day': 20
        },
        'Desktop Shopper': {
            'avg_session_duration': 15.8,
            'avg_events_per_session': 4,
            'total_transactions': 0,
            'total_revenue': 0,
            'engagement_score': 45,
            'age_group': '35-44',
            'gender_clean': 'male',
            'traffic_source_category': 'Paid Traffic',
            'region': 'New York',
            'hour_of_day': 10
        },
        'Returning Customer': {
            'avg_session_duration': 12.3,
            'avg_events_per_session': 6,
            'total_transactions': 0,
            'total_revenue': 0,
            'engagement_score': 60,
            'age_group': '25-34',
            'gender_clean': 'female',
            'traffic_source_category': 'Direct',
            'region': 'Texas',
            'hour_of_day': 16
        }
    }
    
    print("\nGenerating recommendations for different user types:")
    
    for user_type, profile in user_profiles.items():
        print(f"\n👤 {user_type}:")
        recommendations = cold_start.get_cold_start_recommendations(profile)
        final_recs = recommendations['final_recommendations']
        
        print(f"   • Categories: {final_recs['recommended_categories'][:3]}...")
        print(f"   • Engagement: {final_recs['recommended_engagement_level']}")
        print(f"   • Confidence: {final_recs['confidence_score']:.2f}")

def main():
    """
    Main demo function that runs all demonstrations.
    """
    print("🚀 COLD START STRATEGY DEMO")
    print("="*60)
    
    # Check if sample data exists, create if not
    import os
    if not os.path.exists('sample_data/user_segments.csv'):
        print("Sample data not found. Creating sample data...")
        create_sample_data()
    else:
        print("Sample data found. Using existing data...")
    
    # Run basic demo
    cold_start, recommendations = demo_basic_usage()
    
    # Run advanced features demo
    results = demo_advanced_features()
    
    # Run multiple users demo
    demo_multiple_users()
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  • sample_data/ - Sample datasets")
    print("  • cold_start_data/ - Generated models and trends")
    print("\nYou can now use the ColdStartStrategy in your own applications!")

if __name__ == "__main__":
    main() 