import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import os

# Fix for Windows joblib issue
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
warnings.filterwarnings('ignore')

class ColdStartStrategy:
    def __init__(self, user_segments_file: str = 'user_segments/comprehensive_user_segments.csv',
                 sessions_file: str = 'processed_data/user_sessions.csv',
                 merged_file: str = 'processed_data/merged_activity_transactions.csv'):
        """
        Initialize the ColdStartStrategy with user segments and activity data.
        
        Args:
            user_segments_file (str): Path to the user segments CSV file
            sessions_file (str): Path to the sessions CSV file
            merged_file (str): Path to the merged activity-transactions CSV file
        """
        self.user_segments_file = user_segments_file
        self.sessions_file = sessions_file
        self.merged_file = merged_file
        self.user_segments_df: Optional[pd.DataFrame] = None
        self.sessions_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.user_clusters: Optional[pd.DataFrame] = None
        self.knn_model: Optional[NearestNeighbors] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
    def load_data(self):
        """
        Load the required datasets for cold start strategy with memory optimization.
        """
        print("Loading data for cold start strategy...")
        
        try:
            # Load user segments with memory optimization
            print("  Loading user segments...")
            self.user_segments_df = pd.read_csv(self.user_segments_file)
            print(f"  Loaded {len(self.user_segments_df)} user segments")
            
            # Load sessions data with memory optimization
            print("  Loading sessions data...")
            # Check file size and use chunked reading if too large
            sessions_file_size = os.path.getsize(self.sessions_file) / (1024 * 1024)  # Size in MB
            print(f"  Sessions file size: {sessions_file_size:.2f} MB")
            
            if sessions_file_size > 100:  # If larger than 100MB, use chunked reading
                print("  Using chunked reading for large sessions file...")
                chunk_size = 50000
                sessions_chunks = []
                for chunk in pd.read_csv(self.sessions_file, chunksize=chunk_size):
                    sessions_chunks.append(chunk)
                self.sessions_df = pd.concat(sessions_chunks, ignore_index=True)
                del sessions_chunks  # Free memory
            else:
                self.sessions_df = pd.read_csv(self.sessions_file)
            
            print(f"  Loaded {len(self.sessions_df)} sessions")
            
            # Load merged data with memory optimization
            print("  Loading merged data...")
            merged_file_size = os.path.getsize(self.merged_file) / (1024 * 1024)  # Size in MB
            print(f"  Merged file size: {merged_file_size:.2f} MB")
            
            if merged_file_size > 200:  # If larger than 200MB, use chunked reading
                print("  Using chunked reading for large merged file...")
                chunk_size = 100000
                merged_chunks = []
                for chunk in pd.read_csv(self.merged_file, chunksize=chunk_size):
                    merged_chunks.append(chunk)
                self.merged_df = pd.concat(merged_chunks, ignore_index=True)
                del merged_chunks  # Free memory
            else:
                self.merged_df = pd.read_csv(self.merged_file)
            
            print(f"  Loaded {len(self.merged_df)} merged records")
            
            # Convert date columns with error handling
            print("  Converting date columns...")
            if 'session_start' in self.sessions_df.columns:
                try:
                    self.sessions_df['session_start'] = pd.to_datetime(self.sessions_df['session_start'], errors='coerce')
                except Exception as e:
                    print(f"    Warning: Could not convert session_start: {e}")
                    
            if 'session_end' in self.sessions_df.columns:
                try:
                    self.sessions_df['session_end'] = pd.to_datetime(self.sessions_df['session_end'], errors='coerce')
                except Exception as e:
                    print(f"    Warning: Could not convert session_end: {e}")
            
            # Memory optimization: drop unnecessary columns early
            print("  Optimizing memory usage...")
            if len(self.sessions_df) > 100000:
                # Keep only essential columns for large datasets
                essential_session_cols = ['user_pseudo_id', 'session_start', 'session_end', 'session_category']
                available_cols = [col for col in essential_session_cols if col in self.sessions_df.columns]
                self.sessions_df = self.sessions_df[available_cols].copy()
                print(f"    Kept {len(available_cols)} essential session columns")
            
            if len(self.merged_df) > 500000:
                # Keep only essential columns for large datasets
                essential_merged_cols = ['user_pseudo_id', 'category', 'city', 'region', 'country', 'source', 'medium']
                available_cols = [col for col in essential_merged_cols if col in self.merged_df.columns]
                self.merged_df = self.merged_df[available_cols].copy()
                print(f"    Kept {len(available_cols)} essential merged columns")
            
            return self
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run user_segmentation.py first to generate the required files.")
            return None
        except MemoryError as e:
            print(f"Memory error during data loading: {e}")
            print("Trying with smaller sample...")
            return self._load_data_with_sampling()
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _load_data_with_sampling(self):
        """
        Load data with sampling to handle memory constraints.
        """
        print("Loading data with sampling due to memory constraints...")
        
        try:
            # Load user segments (usually smaller)
            self.user_segments_df = pd.read_csv(self.user_segments_file)
            print(f"Loaded {len(self.user_segments_df)} user segments")
            
            # Sample sessions data
            print("Sampling sessions data...")
            sample_size = min(50000, len(pd.read_csv(self.sessions_file, nrows=1)))
            self.sessions_df = pd.read_csv(self.sessions_file, nrows=sample_size)
            print(f"Loaded {len(self.sessions_df)} session samples")
            
            # Sample merged data
            print("Sampling merged data...")
            sample_size = min(100000, len(pd.read_csv(self.merged_file, nrows=1)))
            self.merged_df = pd.read_csv(self.merged_file, nrows=sample_size)
            print(f"Loaded {len(self.merged_df)} merged record samples")
            
            # Convert date columns
            if 'session_start' in self.sessions_df.columns:
                self.sessions_df['session_start'] = pd.to_datetime(self.sessions_df['session_start'], errors='coerce')
            if 'session_end' in self.sessions_df.columns:
                self.sessions_df['session_end'] = pd.to_datetime(self.sessions_df['session_end'], errors='coerce')
            
            return self
            
        except Exception as e:
            print(f"Error in sampled loading: {e}")
            return None
    
    def create_user_clusters(self, n_clusters: int = 10):
        """
        Create user clusters using K-means clustering for similar user identification.
        
        Args:
            n_clusters (int): Number of clusters to create
        """
        if self.user_segments_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print(f"\nCreating user clusters (n_clusters={n_clusters})...")
        
        # Select features for clustering
        clustering_features = [
            'avg_session_duration', 'avg_events_per_session', 'total_transactions',
            'total_revenue', 'engagement_score'
        ]
        
        # Prepare data for clustering
        cluster_data = self.user_segments_df[clustering_features].copy()
        
        # Handle missing values
        cluster_data = cluster_data.fillna(cluster_data.mean())
        
        # Scale the features
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(cluster_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster information to user segments
        self.user_segments_df['cluster_id'] = cluster_labels
        
        # Create cluster profiles
        cluster_profiles = self.user_segments_df.groupby('cluster_id').agg({
            'avg_session_duration': 'mean',
            'avg_events_per_session': 'mean',
            'total_transactions': 'mean',
            'total_revenue': 'mean',
            'engagement_score': 'mean',
            'engagement_segment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'customer_segment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'age_group': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'gender_clean': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'traffic_source_category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        # Add cluster size as a separate column
        cluster_sizes = self.user_segments_df.groupby('cluster_id').size()
        cluster_profiles = cluster_profiles.assign(cluster_size=cluster_sizes)
        
        self.user_clusters = cluster_profiles
        
        print(f"Created {n_clusters} user clusters")
        print("Cluster sizes:")
        for cluster_id in cluster_profiles.index:
            size = cluster_profiles.loc[cluster_id, 'cluster_size']
            print(f"  Cluster {cluster_id}: {size} users")
        
        return cluster_profiles
    
    def build_knn_model(self, n_neighbors: int = 5):
        """
        Build KNN model for finding similar users based on profile features.
        
        Args:
            n_neighbors (int): Number of neighbors to consider
        """
        if self.user_segments_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print(f"\nBuilding KNN model (n_neighbors={n_neighbors})...")
        
        # Select features for KNN
        knn_features = [
            'avg_session_duration', 'avg_events_per_session', 'total_transactions',
            'total_revenue', 'engagement_score'
        ]
        
        # Prepare data for KNN
        knn_data = self.user_segments_df[knn_features].copy()
        
        # Handle missing values
        knn_data = knn_data.fillna(knn_data.mean())
        
        # Scale the features
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(knn_data)
        else:
            scaled_data = self.scaler.transform(knn_data)
        
        # Build KNN model
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        self.knn_model.fit(scaled_data)
        
        print("KNN model built successfully")
        return self.knn_model
    
    def find_similar_users(self, user_profile: Dict[str, Any], n_neighbors: int = 5) -> List[Dict]:
        """
        Find similar users using KNN based on user profile.
        
        Args:
            user_profile (Dict): User profile with features
            n_neighbors (int): Number of similar users to find
            
        Returns:
            List[Dict]: List of similar user profiles
        """
        if self.knn_model is None or self.scaler is None:
            raise ValueError("KNN model or scaler not built. Please call build_knn_model() first.")
        
        # Extract features from user profile
        features = [
            user_profile.get('avg_session_duration', 0),
            user_profile.get('avg_events_per_session', 0),
            user_profile.get('total_transactions', 0),
            user_profile.get('total_revenue', 0),
            user_profile.get('engagement_score', 0)
        ]
        
        # Scale the features
        features_scaled = self.scaler.transform([features])
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(features_scaled)
        
        # Get similar user profiles
        similar_users = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if self.user_segments_df is not None:
                similar_user = self.user_segments_df.iloc[idx].to_dict()
                similar_user['similarity_score'] = 1 / (1 + distance)  # Convert distance to similarity
                similar_users.append(similar_user)
        
        return similar_users
    
    def get_default_category_trends(self) -> Dict[str, Dict]:
        """
        Get default category trends by region, time of day, and device.
        Memory-optimized version to handle large datasets.
        
        Returns:
            Dict: Default category trends
        """
        if self.merged_df is None or self.sessions_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print("\nAnalyzing default category trends...")
        
        # Memory optimization: Only select necessary columns and sample if too large
        print("  Optimizing data for memory efficiency...")
        
        # Check data sizes and sample if necessary
        sessions_sample_size = min(100000, len(self.sessions_df))
        merged_sample_size = min(500000, len(self.merged_df))
        
        if len(self.sessions_df) > sessions_sample_size:
            print(f"  Sampling {sessions_sample_size} sessions from {len(self.sessions_df)} total")
            sessions_sample = self.sessions_df.sample(n=sessions_sample_size, random_state=42)
        else:
            sessions_sample = self.sessions_df
            
        if len(self.merged_df) > merged_sample_size:
            print(f"  Sampling {merged_sample_size} merged records from {len(self.merged_df)} total")
            merged_sample = self.merged_df.sample(n=merged_sample_size, random_state=42)
        else:
            merged_sample = self.merged_df
        
        # Select only necessary columns to reduce memory usage
        sessions_cols = ['user_pseudo_id', 'session_start', 'session_end', 'session_category']
        merged_cols = ['user_pseudo_id', 'category', 'city', 'region', 'country', 'source', 'medium']
        
        sessions_subset = sessions_sample[sessions_cols].copy()
        merged_subset = merged_sample[merged_cols].copy()
        
        # Add time-based features to sessions data
        print("  Adding time-based features...")
        sessions_subset['session_start'] = pd.to_datetime(sessions_subset['session_start'])
        sessions_subset['hour_of_day'] = sessions_subset['session_start'].dt.hour
        sessions_subset['day_of_week'] = sessions_subset['session_start'].dt.day_name()
        
        # Define time periods
        def categorize_time_of_day(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
        
        sessions_subset['time_of_day'] = sessions_subset['hour_of_day'].apply(categorize_time_of_day)
        
        # Memory-efficient merge: use smaller chunks
        print("  Performing memory-efficient merge...")
        try:
            # Try direct merge first
            analysis_data = sessions_subset.merge(
                merged_subset,
                on='user_pseudo_id',
                how='left'
            )
        except MemoryError:
            print("  Memory error in merge, using chunked processing...")
            # If memory error, process in chunks
            analysis_data = self._chunked_merge(sessions_subset, merged_subset)
        
        # Clean up memory
        del sessions_subset, merged_subset
        
        trends = {}
        
        # 1. Regional trends - process in smaller groups
        print("  Analyzing regional trends...")
        try:
            regional_trends = analysis_data.groupby(['region', 'category']).agg({
                'session_category': 'count',
            }).reset_index()
            
            # Add revenue if available (might not be in all datasets)
            if 'total_revenue' in analysis_data.columns:
                revenue_trends = analysis_data.groupby(['region', 'category'])['total_revenue'].sum().reset_index()
                regional_trends = regional_trends.merge(revenue_trends, on=['region', 'category'], how='left')
            else:
                regional_trends['total_revenue'] = 0
            
            # Rename columns to avoid confusion
            regional_trends = regional_trends.rename(columns={'session_category': 'session_count'})
            
            trends['regional'] = {}
            for region in regional_trends['region'].unique():
                region_data = regional_trends[regional_trends['region'] == region]
                # Sort by session count and get top 5
                region_data_sorted = region_data.sort_values('session_count', ascending=False)
                top_categories = region_data_sorted.head(5)[['category', 'session_count']]
                # Sort by revenue and get top 5
                region_data_sorted_revenue = region_data.sort_values('total_revenue', ascending=False)
                top_revenue = region_data_sorted_revenue.head(5)[['category', 'total_revenue']]
                
                trends['regional'][region] = {
                    'top_categories': top_categories.to_dict(orient='records'),
                    'top_revenue_categories': top_revenue.to_dict(orient='records')
                }
        except Exception as e:
            print(f"  Error in regional trends: {e}")
            trends['regional'] = {}
        
        # 2. Time of day trends
        print("  Analyzing time of day trends...")
        try:
            time_trends = analysis_data.groupby(['time_of_day', 'category']).agg({
                'session_category': 'count',
            }).reset_index()
            
            # Add revenue if available
            if 'total_revenue' in analysis_data.columns:
                revenue_time_trends = analysis_data.groupby(['time_of_day', 'category'])['total_revenue'].sum().reset_index()
                time_trends = time_trends.merge(revenue_time_trends, on=['time_of_day', 'category'], how='left')
            else:
                time_trends['total_revenue'] = 0
            
            # Rename columns to avoid confusion
            time_trends = time_trends.rename(columns={'session_category': 'session_count'})
            
            trends['time_of_day'] = {}
            for time_period in time_trends['time_of_day'].unique():
                time_data = time_trends[time_trends['time_of_day'] == time_period]
                # Sort by session count and get top 5
                time_data_sorted = time_data.sort_values('session_count', ascending=False)
                top_categories = time_data_sorted.head(5)[['category', 'session_count']]
                # Sort by revenue and get top 5
                time_data_sorted_revenue = time_data.sort_values('total_revenue', ascending=False)
                top_revenue = time_data_sorted_revenue.head(5)[['category', 'total_revenue']]
                
                trends['time_of_day'][time_period] = {
                    'top_categories': top_categories.to_dict(orient='records'),
                    'top_revenue_categories': top_revenue.to_dict(orient='records')
                }
        except Exception as e:
            print(f"  Error in time of day trends: {e}")
            trends['time_of_day'] = {}
        
        # 3. Device/Source trends
        print("  Analyzing device/source trends...")
        try:
            source_trends = analysis_data.groupby(['source', 'category']).agg({
                'session_category': 'count',
            }).reset_index()
            
            # Add revenue if available
            if 'total_revenue' in analysis_data.columns:
                revenue_source_trends = analysis_data.groupby(['source', 'category'])['total_revenue'].sum().reset_index()
                source_trends = source_trends.merge(revenue_source_trends, on=['source', 'category'], how='left')
            else:
                source_trends['total_revenue'] = 0
            
            # Rename columns to avoid confusion
            source_trends = source_trends.rename(columns={'session_category': 'session_count'})
            
            trends['source'] = {}
            for source in source_trends['source'].unique():
                source_data = source_trends[source_trends['source'] == source]
                # Sort by session count and get top 5
                source_data_sorted = source_data.sort_values('session_count', ascending=False)
                top_categories = source_data_sorted.head(5)[['category', 'session_count']]
                # Sort by revenue and get top 5
                source_data_sorted_revenue = source_data.sort_values('total_revenue', ascending=False)
                top_revenue = source_data_sorted_revenue.head(5)[['category', 'total_revenue']]
                
                trends['source'][source] = {
                    'top_categories': top_categories.to_dict(orient='records'),
                    'top_revenue_categories': top_revenue.to_dict(orient='records')
                }
        except Exception as e:
            print(f"  Error in source trends: {e}")
            trends['source'] = {}
        
        # Clean up memory
        del analysis_data
        
        return trends
    
    def _chunked_merge(self, sessions_df: pd.DataFrame, merged_df: pd.DataFrame, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Perform merge in chunks to avoid memory issues.
        
        Args:
            sessions_df: Sessions dataframe
            merged_df: Merged dataframe
            chunk_size: Size of chunks to process
            
        Returns:
            Merged dataframe
        """
        print(f"    Processing merge in chunks of {chunk_size}...")
        
        # Split sessions into chunks
        session_chunks = [sessions_df[i:i+chunk_size] for i in range(0, len(sessions_df), chunk_size)]
        
        merged_chunks = []
        for i, chunk in enumerate(session_chunks):
            print(f"    Processing chunk {i+1}/{len(session_chunks)}...")
            try:
                merged_chunk = chunk.merge(merged_df, on='user_pseudo_id', how='left')
                merged_chunks.append(merged_chunk)
            except Exception as e:
                print(f"    Error in chunk {i+1}: {e}")
                # If merge fails, just use the chunk as is
                merged_chunks.append(chunk)
        
        # Combine all chunks
        if merged_chunks:
            return pd.concat(merged_chunks, ignore_index=True)
        else:
            return sessions_df
    
    def demographic_filtering(self, user_demographics: Dict[str, Any]) -> List[Dict]:
        """
        Find users with similar demographics using filtering.
        
        Args:
            user_demographics (Dict): User demographic information
            
        Returns:
            List[Dict]: List of users with similar demographics
        """
        if self.user_segments_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print(f"\nFinding users with similar demographics...")
        
        # Create filter conditions
        filters = []
        
        if 'age_group' in user_demographics and user_demographics['age_group'] != 'Unknown':
            filters.append(f"age_group == '{user_demographics['age_group']}'")
        
        if 'gender_clean' in user_demographics and user_demographics['gender_clean'] != 'unknown':
            filters.append(f"gender_clean == '{user_demographics['gender_clean']}'")
        
        if 'income_bracket' in user_demographics and user_demographics['income_bracket'] != 'Unknown':
            filters.append(f"income_bracket == '{user_demographics['income_bracket']}'")
        
        if 'traffic_source_category' in user_demographics:
            filters.append(f"traffic_source_category == '{user_demographics['traffic_source_category']}'")
        
        # Apply filters
        if filters:
            filter_query = ' and '.join(filters)
            similar_users = self.user_segments_df.query(filter_query)
        else:
            similar_users = self.user_segments_df
        
        # Sort by engagement score and limit results
        similar_users = similar_users.nlargest(20, 'engagement_score')
        
        print(f"Found {len(similar_users)} users with similar demographics")
        
        return similar_users.to_dict('records')
    
    def rule_based_recommendations(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate rule-based recommendations based on user context.
        
        Args:
            user_context (Dict): User context information
            
        Returns:
            Dict: Rule-based recommendations
        """
        print(f"\nGenerating rule-based recommendations...")
        
        recommendations = {
            'recommended_categories': [],
            'recommended_engagement_level': 'Medium',
            'recommended_content_type': 'General',
            'confidence_score': 0.5
        }
        
        # Rule 1: Time-based recommendations
        hour = user_context.get('hour_of_day', 12)
        if 6 <= hour < 12:
            recommendations['recommended_content_type'] = 'Morning'
            recommendations['confidence_score'] += 0.1
        elif 12 <= hour < 17:
            recommendations['recommended_content_type'] = 'Afternoon'
            recommendations['confidence_score'] += 0.1
        elif 17 <= hour < 21:
            recommendations['recommended_content_type'] = 'Evening'
            recommendations['confidence_score'] += 0.1
        else:
            recommendations['recommended_content_type'] = 'Night'
            recommendations['confidence_score'] += 0.1
        
        # Rule 2: Device/Source based recommendations
        source = user_context.get('source', 'unknown').lower()
        if 'mobile' in source or 'android' in source or 'ios' in source:
            recommendations['recommended_engagement_level'] = 'High'
            recommendations['confidence_score'] += 0.1
        elif 'desktop' in source:
            recommendations['recommended_engagement_level'] = 'Medium'
            recommendations['confidence_score'] += 0.1
        
        # Rule 3: Geographic recommendations
        region = user_context.get('region', 'Unknown')
        if region != 'Unknown':
            recommendations['confidence_score'] += 0.1
        
        # Rule 4: Traffic source recommendations
        traffic_source = user_context.get('traffic_source_category', 'Other')
        if traffic_source == 'Social Media':
            recommendations['recommended_engagement_level'] = 'High'
            recommendations['confidence_score'] += 0.1
        elif traffic_source == 'Paid Traffic':
            recommendations['recommended_engagement_level'] = 'Medium'
            recommendations['confidence_score'] += 0.1
        
        # Cap confidence score at 1.0
        recommendations['confidence_score'] = min(recommendations['confidence_score'], 1.0)
        
        return recommendations
    
    def get_simple_default_trends(self) -> Dict[str, Dict]:
        """
        Generate simple default trends without heavy data processing.
        Fallback method for when memory is constrained.
        
        Returns:
            Dict: Simple default category trends
        """
        print("\nGenerating simple default trends (memory-efficient fallback)...")
        
        # Create basic default trends based on common e-commerce patterns
        trends = {
            'regional': {
                'Unknown': {
                    'top_categories': [
                        {'category': 'Clothing', 'session_count': 1000},
                        {'category': 'Shoes', 'session_count': 800},
                        {'category': 'Bags', 'session_count': 600},
                        {'category': 'Accessories', 'session_count': 400},
                        {'category': 'Jewelry', 'session_count': 300}
                    ],
                    'top_revenue_categories': [
                        {'category': 'Clothing', 'total_revenue': 50000},
                        {'category': 'Shoes', 'total_revenue': 40000},
                        {'category': 'Bags', 'total_revenue': 30000},
                        {'category': 'Accessories', 'total_revenue': 20000},
                        {'category': 'Jewelry', 'total_revenue': 15000}
                    ]
                }
            },
            'time_of_day': {
                'Morning': {
                    'top_categories': [
                        {'category': 'Clothing', 'session_count': 300},
                        {'category': 'Shoes', 'session_count': 250},
                        {'category': 'Bags', 'session_count': 200}
                    ],
                    'top_revenue_categories': [
                        {'category': 'Clothing', 'total_revenue': 15000},
                        {'category': 'Shoes', 'total_revenue': 12000},
                        {'category': 'Bags', 'total_revenue': 10000}
                    ]
                },
                'Afternoon': {
                    'top_categories': [
                        {'category': 'Clothing', 'session_count': 400},
                        {'category': 'Shoes', 'session_count': 350},
                        {'category': 'Bags', 'session_count': 300}
                    ],
                    'top_revenue_categories': [
                        {'category': 'Clothing', 'total_revenue': 20000},
                        {'category': 'Shoes', 'total_revenue': 18000},
                        {'category': 'Bags', 'total_revenue': 15000}
                    ]
                },
                'Evening': {
                    'top_categories': [
                        {'category': 'Clothing', 'session_count': 500},
                        {'category': 'Shoes', 'session_count': 400},
                        {'category': 'Bags', 'session_count': 350}
                    ],
                    'top_revenue_categories': [
                        {'category': 'Clothing', 'total_revenue': 25000},
                        {'category': 'Shoes', 'total_revenue': 20000},
                        {'category': 'Bags', 'total_revenue': 18000}
                    ]
                },
                'Night': {
                    'top_categories': [
                        {'category': 'Clothing', 'session_count': 200},
                        {'category': 'Shoes', 'session_count': 150},
                        {'category': 'Bags', 'session_count': 100}
                    ],
                    'top_revenue_categories': [
                        {'category': 'Clothing', 'total_revenue': 10000},
                        {'category': 'Shoes', 'total_revenue': 8000},
                        {'category': 'Bags', 'total_revenue': 5000}
                    ]
                }
            },
            'source': {
                'google': {
                    'top_categories': [
                        {'category': 'Clothing', 'session_count': 600},
                        {'category': 'Shoes', 'session_count': 500},
                        {'category': 'Bags', 'session_count': 400}
                    ],
                    'top_revenue_categories': [
                        {'category': 'Clothing', 'total_revenue': 30000},
                        {'category': 'Shoes', 'total_revenue': 25000},
                        {'category': 'Bags', 'total_revenue': 20000}
                    ]
                },
                'facebook': {
                    'top_categories': [
                        {'category': 'Clothing', 'session_count': 400},
                        {'category': 'Shoes', 'session_count': 300},
                        {'category': 'Bags', 'session_count': 250}
                    ],
                    'top_revenue_categories': [
                        {'category': 'Clothing', 'total_revenue': 20000},
                        {'category': 'Shoes', 'total_revenue': 15000},
                        {'category': 'Bags', 'total_revenue': 12000}
                    ]
                },
                'direct': {
                    'top_categories': [
                        {'category': 'Clothing', 'session_count': 300},
                        {'category': 'Shoes', 'session_count': 250},
                        {'category': 'Bags', 'session_count': 200}
                    ],
                    'top_revenue_categories': [
                        {'category': 'Clothing', 'total_revenue': 15000},
                        {'category': 'Shoes', 'total_revenue': 12000},
                        {'category': 'Bags', 'total_revenue': 10000}
                    ]
                }
            }
        }
        
        return trends
    
    def get_cold_start_recommendations(self, new_user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive cold start recommendations using multiple strategies.
        
        Args:
            new_user_profile (Dict): Profile of the new user
            
        Returns:
            Dict: Comprehensive cold start recommendations
        """
        print("\n" + "="*60)
        print("COLD START RECOMMENDATION GENERATION")
        print("="*60)
        
        recommendations = {
            'similar_users': [],
            'default_trends': {},
            'demographic_matches': [],
            'rule_based': {},
            'final_recommendations': {}
        }
        
        # Strategy 1: Find similar users using KNN
        print("\n1. Finding similar users (KNN)...")
        try:
            similar_users = self.find_similar_users(new_user_profile, n_neighbors=5)
            recommendations['similar_users'] = similar_users
            print(f"   Found {len(similar_users)} similar users")
        except Exception as e:
            print(f"   Error in KNN: {e}")
        
        # Strategy 2: Get default category trends
        print("\n2. Getting default category trends...")
        try:
            default_trends = self.get_default_category_trends()
            recommendations['default_trends'] = default_trends
            print("   Default trends generated successfully")
        except Exception as e:
            print(f"   Error in default trends: {e}")
        
        # Strategy 3: Demographic filtering
        print("\n3. Demographic filtering...")
        try:
            demographic_matches = self.demographic_filtering(new_user_profile)
            recommendations['demographic_matches'] = demographic_matches
            print(f"   Found {len(demographic_matches)} demographic matches")
        except Exception as e:
            print(f"   Error in demographic filtering: {e}")
        
        # Strategy 4: Rule-based recommendations
        print("\n4. Rule-based recommendations...")
        try:
            rule_based = self.rule_based_recommendations(new_user_profile)
            recommendations['rule_based'] = rule_based
            print("   Rule-based recommendations generated")
        except Exception as e:
            print(f"   Error in rule-based recommendations: {e}")
        
        # Generate final recommendations
        recommendations['final_recommendations'] = self._combine_recommendations(recommendations)
        
        return recommendations
    
    def _combine_recommendations(self, all_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine recommendations from different strategies into final recommendations.
        
        Args:
            all_recommendations (Dict): All recommendations from different strategies
            
        Returns:
            Dict: Combined final recommendations
        """
        print("\n5. Combining recommendations...")
        
        final_recommendations = {
            'recommended_categories': [],
            'recommended_engagement_level': 'Medium',
            'recommended_content_type': 'General',
            'confidence_score': 0.5,
            'fallback_strategy': 'default_trends'
        }
        
        # Combine similar user preferences
        if all_recommendations['similar_users']:
            similar_user_categories = []
            for user in all_recommendations['similar_users']:
                if 'session_categories' in user:
                    similar_user_categories.extend(user['session_categories'])
            
            if similar_user_categories:
                category_counts = pd.Series(similar_user_categories).value_counts()
                final_recommendations['recommended_categories'] = category_counts.head(5).index.tolist()
                final_recommendations['confidence_score'] += 0.2
                final_recommendations['fallback_strategy'] = 'similar_users'
        
        # Use default trends if no similar users
        if not final_recommendations['recommended_categories'] and all_recommendations['default_trends']:
            region = 'Unknown'  # Default region
            if 'regional' in all_recommendations['default_trends'] and region in all_recommendations['default_trends']['regional']:
                top_categories = all_recommendations['default_trends']['regional'][region]['top_categories']
                final_recommendations['recommended_categories'] = [cat['category'] for cat in top_categories[:5]]
                final_recommendations['confidence_score'] += 0.1
        
        # Combine engagement level recommendations
        engagement_scores = []
        if all_recommendations['similar_users']:
            engagement_scores.extend([user.get('engagement_score', 50) for user in all_recommendations['similar_users']])
        
        if engagement_scores:
            avg_engagement = np.mean(engagement_scores)
            if avg_engagement >= 70:
                final_recommendations['recommended_engagement_level'] = 'High'
            elif avg_engagement >= 40:
                final_recommendations['recommended_engagement_level'] = 'Medium'
            else:
                final_recommendations['recommended_engagement_level'] = 'Low'
        
        # Use rule-based recommendations
        if all_recommendations['rule_based']:
            final_recommendations['recommended_content_type'] = all_recommendations['rule_based'].get('recommended_content_type', 'General')
            final_recommendations['confidence_score'] = max(
                final_recommendations['confidence_score'],
                all_recommendations['rule_based'].get('confidence_score', 0.5)
            )
        
        # Cap confidence score
        final_recommendations['confidence_score'] = min(final_recommendations['confidence_score'], 1.0)
        
        return final_recommendations
    
    def save_cold_start_data(self, output_dir: str = 'cold_start_data'):
        """
        Save cold start strategy data and models.
        
        Args:
            output_dir (str): Directory to save cold start data
        """
        import os
        import pickle
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving cold start data to {output_dir}...")
        
        # Save user clusters
        if self.user_clusters is not None:
            self.user_clusters.to_csv(f'{output_dir}/user_clusters.csv')
            print(f"Saved user clusters: {output_dir}/user_clusters.csv")
        
        # Save KNN model and scaler
        if self.knn_model is not None and self.scaler is not None:
            with open(f'{output_dir}/knn_model.pkl', 'wb') as f:
                pickle.dump(self.knn_model, f)
            with open(f'{output_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Saved KNN model and scaler: {output_dir}/")
        
        # Save default trends
        default_trends = self.get_default_category_trends()
        import json
        with open(f'{output_dir}/default_trends.json', 'w') as f:
            json.dump(default_trends, f, indent=2)
        print(f"Saved default trends: {output_dir}/default_trends.json")
        
        return self
    
    def run_full_cold_start_pipeline(self, save_data: bool = True):
        """
        Run the complete cold start strategy pipeline with memory optimization.
        
        Args:
            save_data (bool): Whether to save cold start data
        """
        print("Starting Cold Start Strategy Pipeline")
        print("=" * 50)
        
        try:
            # Load data
            if self.load_data() is None:
                return None
            
            # Create user clusters
            self.create_user_clusters(n_clusters=10)
            
            # Build KNN model
            self.build_knn_model(n_neighbors=5)
            
            # Generate default trends with memory optimization
            print("\nGenerating default trends...")
            try:
                default_trends = self.get_default_category_trends()
                print("Default trends generated successfully")
            except MemoryError as e:
                print(f"Memory error in default trends: {e}")
                print("Using simple default trends instead...")
                default_trends = self.get_simple_default_trends()
            except Exception as e:
                print(f"Error in default trends: {e}")
                print("Using simple default trends instead...")
                default_trends = self.get_simple_default_trends()
            
            # Save data if requested
            if save_data:
                self.save_cold_start_data()
            
            print("\nCold Start Strategy Pipeline Completed Successfully!")
            print("=" * 50)
            
            return {
                'user_clusters': self.user_clusters,
                'default_trends': default_trends,
                'knn_model': self.knn_model
            }
            
        except Exception as e:
            print(f"Error in cold start pipeline: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize cold start strategy
    cold_start = ColdStartStrategy()
    
    # Run the full pipeline
    results = cold_start.run_full_cold_start_pipeline(save_data=True)
    
    if results:
        print(f"\nCold start strategy completed successfully!")
        print(f"Created {len(results['user_clusters'])} user clusters")
        
        # Example: Generate recommendations for a new user
        new_user_profile = {
            'avg_session_duration': 8.5,
            'avg_events_per_session': 6,
            'total_transactions': 0,
            'total_revenue': 0,
            'engagement_score': 45,
            'age_group': '25-34',
            'gender_clean': 'female',
            'traffic_source_category': 'Social Media',
            'region': 'California',
            'hour_of_day': 14
        }
        
        recommendations = cold_start.get_cold_start_recommendations(new_user_profile)
        
        print(f"\nExample recommendations for new user:")
        print(f"Recommended categories: {recommendations['final_recommendations']['recommended_categories']}")
        print(f"Recommended engagement level: {recommendations['final_recommendations']['recommended_engagement_level']}")
        print(f"Confidence score: {recommendations['final_recommendations']['confidence_score']:.2f}")
        print(f"Fallback strategy: {recommendations['final_recommendations']['fallback_strategy']}") 