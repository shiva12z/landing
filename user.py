```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional
import uvicorn
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hyper-Personalized Landing Page Generator")

# Enable CORS for front-end integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for context input
class Context(BaseModel):
    region: Optional[str] = None
    device_type: Optional[str] = None
    traffic_source: Optional[str] = None

# Load and preprocess data
def load_data(csv_path: str = "merged_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found")
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    
    try:
        df = pd.read_csv(csv_path)
        # Ensure consistent column names and handle case sensitivity
        df.columns = df.columns.str.lower()
        if 'transaction_id' in df.columns:
            df['transaction_id'] = df['transaction_id'].astype(str)
        df.fillna({
            'session_duration': 0,
            'product_views': 0,
            'purchases': 0,
            'region': 'unknown',
            'device_type': 'unknown',
            'traffic_source': 'unknown',
            'age_group': 'unknown'
        }, inplace=True)
        logger.info("Data loaded and preprocessed successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load data")

# Feature engineering for user segmentation
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Aggregate user-level features
        user_features = df.groupby('user_pseudo_id').agg({
            'session_duration': 'mean',
            'product_views': 'sum',
            'purchases': 'sum',
            'region': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'device_type': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'traffic_source': lambda x: x.mode()[0] if not x.empty else 'unknown',
            'age_group': lambda x: x.mode()[0] if not x.empty else 'unknown'
        }).reset_index()
        
        # Derived features
        user_features['view_to_purchase_ratio'] = user_features['product_views'] / (user_features['purchases'] + 1)
        user_features['engagement_score'] = user_features['session_duration'] * user_features['product_views']
        
        # Encode categorical variables
        categorical_cols = ['region', 'device_type', 'traffic_source', 'age_group']
        user_features = pd.get_dummies(user_features, columns=categorical_cols, drop_first=True)
        
        logger.info("Feature engineering completed")
        return user_features
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to engineer features")

# Segment users using K-means clustering
def segment_users(user_features: pd.DataFrame, n_clusters: int = 5) -> tuple:
    try:
        feature_cols = [col for col in user_features.columns if col != 'user_pseudo_id']
        X = user_features[feature_cols]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        user_features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Evaluate clustering
        silhouette = silhouette_score(X_scaled, user_features['cluster'])
        logger.info(f"Silhouette Score for {n_clusters} clusters: {silhouette:.2f}")
        
        # Save models for persistence
        joblib.dump(kmeans, 'kmeans_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        user_features.to_csv('user_segments.csv', index=False)
        
        return user_features, kmeans, scaler
    except Exception as e:
        logger.error(f"Error in user segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to segment users")

# Map clusters to landing page modules
def map_clusters_to_modules():
    return {
        0: {
            'hero_image': 'premium_products.jpg',
            'product_carousel': ['Luxury Watch', 'Designer Bag'],
            'cta': 'Shop Premium Now'
        },
        1: {
            'hero_image': 'discount_deals.jpg',
            'product_carousel': ['Budget Phone', 'Clearance Shoes'],
            'cta': 'Grab Deals Today'
        },
        2: {
            'hero_image': 'trending_fashion.jpg',
            'product_carousel': ['T-Shirt', 'Sneakers'],
            'cta': 'Explore Trends'
        },
        3: {
            'hero_image': 'home_essentials.jpg',
            'product_carousel': ['Sofa', 'Lamp'],
            'cta': 'Upgrade Your Home'
        },
        4: {
            'hero_image': 'tech_gadgets.jpg',
            'product_carousel': ['Smartphone', 'Headphones'],
            'cta': 'Discover Tech'
        }
    }

# Cold start strategy for new users
def cold_start_recommendation(context: dict) -> dict:
    region = context.get('region', 'unknown')
    device = context.get('device_type', 'unknown')
    traffic = context.get('traffic_source', 'unknown')
    
    try:
        if region in ['urban', 'metro']:
            return {
                'hero_image': 'trending_products.jpg',
                'product_carousel': ['Smartphone', 'Fashion'],
                'cta': 'Shop Top Picks'
            }
        elif device == 'mobile':
            return {
                'hero_image': 'mobile_deals.jpg',
                'product_carousel': ['Accessories', 'Gadgets'],
                'cta': 'Browse on the Go'
            }
        elif traffic == 'social_media':
            return {
                'hero_image': 'viral_items.jpg',
                'product_carousel': ['Trending Shirt', 'Viral Gadget'],
                'cta': 'Get Viral Products'
            }
        else:
            return {
                'hero_image': 'best_sellers.jpg',
                'product_carousel': ['Popular Book', 'Top Headphones'],
                'cta': 'Shop Best Sellers'
            }
    except Exception as e:
        logger.error(f"Error in cold start recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate cold start recommendation")

# Load data and initialize models
try:
    df = load_data()
    user_features = engineer_features(df)
    user_features, kmeans, scaler = segment_users(user_features)
    module_configs = map_clusters_to_modules()
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise Exception("Failed to initialize backend")

# API endpoint for generating personalized landing page
@app.get("/generate_landing_page/{user_id}")
async def generate_landing_page(user_id: str, context: Context = None):
    try:
        context = context.dict() if context else {}
        if user_id in user_features['user_pseudo_id'].values:
            user_data = user_features[user_features['user_pseudo_id'] == user_id]
            cluster = user_data['cluster'].iloc[0]
            modules = module_configs[cluster]
            logger.info(f"Generated landing page for existing user {user_id} in cluster {cluster}")
        else:
            modules = cold_start_recommendation(context)
            logger.info(f"Generated landing page for new user {user_id} using cold start")
        
        return {'user_id': user_id, 'modules': modules}
    except Exception as e:
        logger.error(f"Error generating landing page for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate landing page")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```