import pandas as pd
import os
from typing import Optional, Dict, Any
import logging
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# Placeholder for ML logic (user behavior analysis, recommendations, etc.)

def analyze_user_behavior(session_data):
    # Implement ML logic here
    pass

def recommend_engagement_type(user_pseudo_id: str, data_path: str = "preprocessed_sessions.csv") -> Optional[str]:
    """
    Recommend the most frequent engagement type for a user based on session history.
    Returns 'high', 'medium', 'low', or None if user not found.
    """
    if not os.path.exists(data_path):
        logging.warning(f"Data file {data_path} not found.")
        return None
    df = pd.read_csv(data_path)
    user_sessions = df[df['user_pseudo_id'] == user_pseudo_id]
    if user_sessions.empty or 'engagement_type' not in user_sessions.columns:
        logging.info(f"No sessions or engagement_type for user {user_pseudo_id}.")
        return None
    mode = user_sessions['engagement_type'].mode()
    if mode.empty:
        logging.info(f"No engagement_type mode for user {user_pseudo_id}.")
        return None
    logging.info(f"Recommended engagement_type for user {user_pseudo_id}: {mode.iloc[0]}")
    return mode.iloc[0]

def segment_users(data_path: str = "preprocessed_sessions.csv") -> dict:
    """
    Segment users by engagement and behavioral attributes.
    Returns a dictionary with segment names and user lists/counts.
    """
    if not os.path.exists(data_path):
        logging.warning(f"Data file {data_path} not found.")
        return {}
    df = pd.read_csv(data_path)
    segments = {}
    # Engagement segments
    engagement_groups = df.groupby('user_pseudo_id')['engagement_type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    segments['high_engagement'] = engagement_groups[engagement_groups == 'high'].index.tolist()
    segments['medium_engagement'] = engagement_groups[engagement_groups == 'medium'].index.tolist()
    segments['low_engagement'] = engagement_groups[engagement_groups == 'low'].index.tolist()
    # Behavioral segments
    if 'event_name' in df.columns:
        # Cart abandoners: users with add_to_cart but no purchase
        cart_users = set(df[df['event_name'] == 'add_to_cart']['user_pseudo_id'])
        purchase_users = set(df[df['event_name'] == 'purchase']['user_pseudo_id'])
        segments['cart_abandoners'] = list(cart_users - purchase_users)
        # Frequent viewers: users with many 'view_item' events
        view_counts = df[df['event_name'] == 'view_item'].groupby('user_pseudo_id').size()
        segments['frequent_viewers'] = view_counts[view_counts > view_counts.mean()].index.tolist()
        # Repeat purchasers: users with more than one purchase
        purchase_counts = df[df['event_name'] == 'purchase'].groupby('user_pseudo_id').size()
        segments['repeat_purchasers'] = purchase_counts[purchase_counts > 1].index.tolist()
    # Demographic bins
    if 'age' in df.columns:
        bins = [0, 18, 25, 35, 45, 60, 120]
        labels = ['<18', '18-24', '25-34', '35-44', '45-59', '60+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        segments['age_groups'] = df.groupby('age_group')['user_pseudo_id'].unique().apply(list).to_dict()
    if 'income' in df.columns:
        income_bins = [0, 30000, 60000, 100000, 200000, float('inf')]
        income_labels = ['<30k', '30k-60k', '60k-100k', '100k-200k', '200k+']
        df['income_bracket'] = pd.cut(df['income'], bins=income_bins, labels=income_labels, right=False)
        segments['income_brackets'] = df.groupby('income_bracket')['user_pseudo_id'].unique().apply(list).to_dict()
    # Gender
    if 'gender' in df.columns:
        segments['gender'] = df.groupby('gender')['user_pseudo_id'].unique().apply(list).to_dict()
    # Source
    if 'source' in df.columns:
        segments['source'] = df.groupby('source')['user_pseudo_id'].unique().apply(list).to_dict()
    # Geography
    for geo in ['city', 'state', 'country']:
        if geo in df.columns:
            segments[geo] = df.groupby(geo)['user_pseudo_id'].unique().apply(list).to_dict()
    return segments

def cold_start_recommendation(user_profile: dict = None, data_path: str = "preprocessed_sessions.csv") -> str:
    """
    Recommend engagement type for a new user using advanced fallback logic:
    1. If user_profile and demographics exist, use KNN to find similar users.
    2. If region/time/device is provided, use default trends for that segment.
    3. Otherwise, return the most common engagement type overall.
    """
    if not os.path.exists(data_path):
        return "medium"  # Default fallback
    df = pd.read_csv(data_path)
    user_profile = user_profile or {}
    # 1. KNN on demographics
    demo_cols = [col for col in ['age', 'gender', 'income', 'city', 'state', 'country'] if col in df.columns and col in user_profile]
    if demo_cols:
        demo_df = df.drop_duplicates('user_pseudo_id').set_index('user_pseudo_id')
        demo_df = demo_df[demo_cols + ['engagement_type']].dropna()
        if not demo_df.empty:
            from sklearn.neighbors import NearestNeighbors
            import pandas as pd
            demo_encoded = pd.get_dummies(demo_df[demo_cols])
            user_vec = pd.DataFrame([user_profile], columns=demo_cols)
            user_vec = pd.get_dummies(user_vec).reindex(columns=demo_encoded.columns, fill_value=0)
            nbrs = NearestNeighbors(n_neighbors=3).fit(demo_encoded.values)
            _, idxs = nbrs.kneighbors(user_vec.values)
            neighbor_types = demo_df.iloc[idxs[0]]['engagement_type']
            if not neighbor_types.empty:
                return neighbor_types.mode().iloc[0]
    # 2. Default trends by region, time, device
    for seg in ['city', 'state', 'country', 'device', 'hour']:
        if seg in user_profile and seg in df.columns:
            seg_val = user_profile[seg]
            seg_df = df[df[seg] == seg_val]
            if not seg_df.empty and 'engagement_type' in seg_df.columns:
                mode = seg_df['engagement_type'].mode()
                if not mode.empty:
                    return mode.iloc[0]
    # 2b. Time of day (if eventtimestamp exists)
    if 'hour' in user_profile and 'eventtimestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['eventtimestamp'], errors='coerce').dt.hour
        seg_df = df[df['hour'] == user_profile['hour']]
        if not seg_df.empty and 'engagement_type' in seg_df.columns:
            mode = seg_df['engagement_type'].mode()
            if not mode.empty:
                return mode.iloc[0]
    # 3. Fallback: most common engagement type overall
    if 'engagement_type' in df.columns:
        mode = df['engagement_type'].mode()
        if not mode.empty:
            return mode.iloc[0]
    return "medium"  # Final fallback

def personalize_landing_page(user_pseudo_id: str = None, user_profile: dict = None, data_path: str = "preprocessed_sessions.csv") -> dict:
    """
    Generate a personalized landing page config (hero, products, CTA) based on user or profile, using region, time, device, and trending categories if available.
    """
    # Default content
    content = {
        "hero_banner": "Welcome to our store!",
        "product_modules": ["Top Sellers", "Trending Now"],
        "cta": "Shop Now"
    }
    if not os.path.exists(data_path):
        return content
    df = pd.read_csv(data_path)
    engagement_type = None
    if user_pseudo_id and user_pseudo_id in df['user_pseudo_id'].values:
        user_sessions = df[df['user_pseudo_id'] == user_pseudo_id]
        if 'engagement_type' in user_sessions.columns and not user_sessions['engagement_type'].mode().empty:
            engagement_type = user_sessions['engagement_type'].mode().iloc[0]
    elif user_profile:
        engagement_type = cold_start_recommendation(user_profile, data_path)
    else:
        engagement_type = cold_start_recommendation({}, data_path)
    # Personalization rules by engagement
    if engagement_type == 'high':
        content["hero_banner"] = "Welcome back, valued customer!"
        content["product_modules"] = ["Recommended For You", "Exclusive Deals"]
        content["cta"] = "Buy Again"
    elif engagement_type == 'medium':
        content["hero_banner"] = "Check out what’s new!"
        content["product_modules"] = ["Recently Viewed", "Popular Picks"]
        content["cta"] = "Add to Cart"
    elif engagement_type == 'low':
        content["hero_banner"] = "Discover our bestsellers!"
        content["product_modules"] = ["Top Sellers", "Trending Now"]
        content["cta"] = "Explore Now"
    # Region-based personalization
    for geo in ['city', 'state', 'country']:
        if user_profile and geo in user_profile and geo in df.columns:
            geo_val = user_profile[geo]
            geo_df = df[df[geo] == geo_val]
            if not geo_df.empty and 'event_name' in geo_df.columns:
                top_cat = geo_df['event_name'].value_counts().idxmax()
                content["hero_banner"] = f"Popular in {geo_val}: {top_cat.title()}!"
                content["product_modules"] = [f"Trending in {geo_val}", top_cat.title()]
                break
    # Time-based personalization
    if user_profile and 'hour' in user_profile and 'eventtimestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['eventtimestamp'], errors='coerce').dt.hour
        hour_df = df[df['hour'] == user_profile['hour']]
        if not hour_df.empty and 'event_name' in hour_df.columns:
            top_cat = hour_df['event_name'].value_counts().idxmax()
            content["hero_banner"] = f"Hot this hour: {top_cat.title()}!"
    # Device-based personalization
    if user_profile and 'device' in user_profile and 'device' in df.columns:
        device_val = user_profile['device']
        device_df = df[df['device'] == device_val]
        if not device_df.empty and 'event_name' in device_df.columns:
            top_cat = device_df['event_name'].value_counts().idxmax()
            content["product_modules"] = [f"{top_cat.title()} for {device_val.title()}"]
    # Age cohort personalization
    if user_profile and 'age' in user_profile:
        if user_profile['age'] < 25:
            content["hero_banner"] = "Student Specials Just For You!"
        elif user_profile['age'] > 60:
            content["hero_banner"] = "Senior Offers Available!"
    # Social source CTA
    if user_profile and user_profile.get('source') == 'social':
        content["cta"] = "Share & Save"
    return content
