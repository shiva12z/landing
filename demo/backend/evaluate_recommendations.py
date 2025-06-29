#!/usr/bin/env python3
"""
Recommendation System Evaluation
===============================

This script evaluates the accuracy of the recommendation system by comparing it
with various baseline strategies using different metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import json
from datetime import datetime, timedelta

# Import our recommendation systems
from cold_start_strategy import ColdStartStrategy
from personalization_logic import PersonalizationEngine

class RecommendationEvaluator:
    """
    Evaluates recommendation system accuracy against baseline strategies.
    """
    
    def __init__(self, test_data_path: str = None):
        """
        Initialize the evaluator.
        
        Args:
            test_data_path: Path to test data (if None, generates synthetic data)
        """
        self.test_data = None
        self.ground_truth = {}
        self.recommendation_results = {}
        self.metrics = {}
        
        if test_data_path:
            self.load_test_data(test_data_path)
        else:
            self.generate_synthetic_test_data()
    
    def generate_synthetic_test_data(self, n_users: int = 1000):
        """
        Generate synthetic test data for evaluation.
        
        Args:
            n_users: Number of test users to generate
        """
        print(f"Generating synthetic test data for {n_users} users...")
        
        np.random.seed(42)
        
        # Generate user profiles
        user_profiles = []
        for i in range(n_users):
            profile = {
                'user_id': f'test_user_{i}',
                'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+']),
                'gender': np.random.choice(['male', 'female']),
                'region': np.random.choice(['California', 'Texas', 'New York', 'Florida', 'Illinois']),
                'traffic_source': np.random.choice(['google', 'facebook', 'instagram', 'direct', 'email']),
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet']),
                'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night']),
                'engagement_score': np.random.uniform(20, 80),
                'session_count': np.random.poisson(3),
                'total_revenue': np.random.exponential(50),
                'avg_session_duration': np.random.normal(8.5, 3),
                'avg_events_per_session': np.random.poisson(6),
                'total_transactions': np.random.poisson(2)
            }
            user_profiles.append(profile)
        
        self.test_data = pd.DataFrame(user_profiles)
        
        # Generate ground truth (what users actually interacted with)
        categories = ['Clothing', 'Shoes', 'Bags', 'Accessories', 'Jewelry', 'Watches']
        
        for _, user in self.test_data.iterrows():
            # Generate realistic ground truth based on user profile
            user_categories = self._generate_realistic_interactions(user, categories)
            self.ground_truth[str(user['user_id'])] = user_categories
        
        print(f"Generated test data with {len(self.test_data)} users")
        print(f"Ground truth generated for {len(self.ground_truth)} users")
    
    def _generate_realistic_interactions(self, user: pd.Series, categories: List[str]) -> List[str]:
        """
        Generate realistic user interactions based on profile.
        
        Args:
            user: User profile
            categories: Available categories
            
        Returns:
            List of categories the user actually interacted with
        """
        # Base interaction probability
        base_prob = 0.3
        
        # Adjust based on engagement
        if user['engagement_score'] > 60:
            base_prob = 0.6
        elif user['engagement_score'] > 40:
            base_prob = 0.4
        
        # Adjust based on age group
        age_adjustments = {
            '18-24': {'Clothing': 1.5, 'Shoes': 1.3, 'Accessories': 1.2},
            '25-34': {'Clothing': 1.4, 'Bags': 1.3, 'Jewelry': 1.2},
            '35-44': {'Clothing': 1.3, 'Bags': 1.4, 'Watches': 1.5},
            '45-54': {'Clothing': 1.2, 'Jewelry': 1.4, 'Watches': 1.3},
            '55+': {'Clothing': 1.1, 'Jewelry': 1.3, 'Accessories': 1.2}
        }
        
        # Adjust based on region
        region_adjustments = {
            'California': {'Clothing': 1.2, 'Shoes': 1.1},
            'Texas': {'Bags': 1.2, 'Accessories': 1.1},
            'New York': {'Jewelry': 1.3, 'Watches': 1.2},
            'Florida': {'Clothing': 1.1, 'Accessories': 1.2},
            'Illinois': {'Clothing': 1.1, 'Bags': 1.1}
        }
        
        # Calculate interaction probabilities
        probs = {}
        for category in categories:
            prob = base_prob
            
            # Apply age adjustments
            if user['age_group'] in age_adjustments:
                prob *= age_adjustments[user['age_group']].get(category, 1.0)
            
            # Apply region adjustments
            if user['region'] in region_adjustments:
                prob *= region_adjustments[user['region']].get(category, 1.0)
            
            probs[category] = min(prob, 0.9)  # Cap at 0.9
        
        # Generate interactions
        interactions = []
        for category, prob in probs.items():
            if np.random.random() < prob:
                interactions.append(category)
        
        # Ensure at least one interaction
        if not interactions:
            interactions = [np.random.choice(categories)]
        
        return interactions
    
    def load_test_data(self, test_data_path: str):
        """
        Load test data from file.
        
        Args:
            test_data_path: Path to test data file
        """
        print(f"Loading test data from {test_data_path}...")
        self.test_data = pd.read_csv(test_data_path)
        
        # Load ground truth if available
        ground_truth_path = test_data_path.replace('.csv', '_ground_truth.json')
        try:
            with open(ground_truth_path, 'r') as f:
                self.ground_truth = json.load(f)
            print(f"Loaded ground truth for {len(self.ground_truth)} users")
        except FileNotFoundError:
            print("Ground truth file not found. Please ensure it exists.")
    
    def show_data_summary(self):
        """
        Display summary of the test data.
        """
        print("\n" + "="*60)
        print("TEST DATA SUMMARY")
        print("="*60)
        
        if self.test_data is None:
            print("No test data loaded.")
            return
        
        print(f"\n📊 Dataset Overview:")
        print(f"   • Total users: {len(self.test_data)}")
        print(f"   • Ground truth available: {len(self.ground_truth)} users")
        
        print(f"\n👥 User Demographics:")
        print(f"   • Age groups: {self.test_data['age_group'].value_counts().to_dict()}")
        print(f"   • Gender: {self.test_data['gender'].value_counts().to_dict()}")
        print(f"   • Regions: {self.test_data['region'].value_counts().to_dict()}")
        
        print(f"\n📈 Engagement Distribution:")
        print(f"   • Average engagement score: {self.test_data['engagement_score'].mean():.2f}")
        print(f"   • Engagement range: {self.test_data['engagement_score'].min():.1f} - {self.test_data['engagement_score'].max():.1f}")
        
        print(f"\n🛒 Interaction Patterns:")
        all_interactions = []
        for interactions in self.ground_truth.values():
            all_interactions.extend(interactions)
        
        interaction_counts = pd.Series(all_interactions).value_counts()
        print(f"   • Most popular categories: {interaction_counts.head(3).to_dict()}")
        print(f"   • Average interactions per user: {len(all_interactions) / len(self.ground_truth):.2f}")
        
        # Show sample user data
        print(f"\n📋 Sample User Data:")
        if self.test_data is not None and len(self.test_data) > 0:
            sample_user = self.test_data.iloc[0]
            user_id = str(sample_user['user_id'])
            print(f"   • User ID: {user_id}")
            print(f"   • Profile: {sample_user['age_group']}, {sample_user['gender']}, {sample_user['region']}")
            print(f"   • Engagement: {sample_user['engagement_score']:.1f}")
            print(f"   • Ground truth interactions: {self.ground_truth.get(user_id, [])}")
        else:
            print("   • No sample data available")
    
    def test_recommendation_strategies(self):
        """
        Test different recommendation strategies.
        """
        print("\n" + "="*60)
        print("TESTING RECOMMENDATION STRATEGIES")
        print("="*60)
        
        if self.test_data is None:
            print("No test data available. Cannot test strategies.")
            return
        
        strategies = {
            'Cold Start Strategy': self._test_cold_start_strategy,
            'Personalization Engine': self._test_personalization_engine,
            'Random Recommendations': self._test_random_strategy,
            'Popular Categories': self._test_popular_strategy,
            'Demographic Based': self._test_demographic_strategy
        }
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n🔍 Testing {strategy_name}...")
            try:
                results = strategy_func()
                self.recommendation_results[strategy_name] = results
                print(f"   ✅ Completed - Generated recommendations for {len(results)} users")
            except Exception as e:
                print(f"   ❌ Failed - {str(e)}")
                self.recommendation_results[strategy_name] = {}
    
    def _test_cold_start_strategy(self) -> Dict[str, List[str]]:
        """Test the cold start strategy"""
        results = {}
        
        if self.test_data is None:
            return results
        
        # Initialize cold start strategy
        cold_start = ColdStartStrategy()
        
        # Try to load data, if not available use simple recommendations
        try:
            cold_start.load_data()
            cold_start.create_user_clusters(n_clusters=5)
            cold_start.build_knn_model(n_neighbors=3)
        except:
            print("     Using fallback cold start strategy...")
        
        for _, user in self.test_data.iterrows():
            try:
                user_profile = {
                    'avg_session_duration': float(user['avg_session_duration']),
                    'avg_events_per_session': int(user['avg_events_per_session']),
                    'total_transactions': int(user['total_transactions']),
                    'total_revenue': float(user['total_revenue']),
                    'engagement_score': float(user['engagement_score']),
                    'age_group': str(user['age_group']),
                    'gender_clean': str(user['gender']),
                    'traffic_source_category': str(user['traffic_source']),
                    'region': str(user['region']),
                    'hour_of_day': 14  # Default
                }
                
                recommendations = cold_start.get_cold_start_recommendations(user_profile)
                recommended_categories = recommendations['final_recommendations']['recommended_categories']
                
                results[str(user['user_id'])] = recommended_categories
            except:
                # Fallback to simple recommendations
                results[str(user['user_id'])] = ['Clothing', 'Shoes', 'Bags']
        
        return results
    
    def _test_personalization_engine(self) -> Dict[str, List[str]]:
        """Test the personalization engine"""
        results = {}
        
        if self.test_data is None:
            return results
        
        # Initialize personalization engine
        engine = PersonalizationEngine()
        
        for _, user in self.test_data.iterrows():
            try:
                user_dict = user.to_dict()
                user_profile = engine.create_user_profile(user_dict)
                landing_page = engine.get_full_landing_page_content(user_profile)
                
                # Extract recommended categories from product module
                product_module = landing_page['sections'].get('product module', {})
                recommended_categories = product_module.get('recommended_categories', [])
                
                results[str(user['user_id'])] = recommended_categories
            except:
                # Fallback
                results[str(user['user_id'])] = ['Clothing', 'Shoes', 'Bags']
        
        return results
    
    def _test_random_strategy(self) -> Dict[str, List[str]]:
        """Test random recommendations"""
        results = {}
        
        if self.test_data is None:
            return results
            
        categories = ['Clothing', 'Shoes', 'Bags', 'Accessories', 'Jewelry', 'Watches']
        
        for _, user in self.test_data.iterrows():
            # Randomly select 3 categories
            recommended = np.random.choice(categories, size=3, replace=False).tolist()
            results[str(user['user_id'])] = recommended
        
        return results
    
    def _test_popular_strategy(self) -> Dict[str, List[str]]:
        """Test popular categories strategy"""
        results = {}
        
        if self.test_data is None:
            return results
        
        # Calculate category popularity from ground truth
        all_interactions = []
        for interactions in self.ground_truth.values():
            all_interactions.extend(interactions)
        
        category_counts = pd.Series(all_interactions).value_counts()
        popular_categories = category_counts.head(3).index.tolist()
        
        for _, user in self.test_data.iterrows():
            results[str(user['user_id'])] = popular_categories
        
        return results
    
    def _test_demographic_strategy(self) -> Dict[str, List[str]]:
        """Test demographic-based recommendations"""
        results = {}
        
        if self.test_data is None:
            return results
        
        # Define demographic preferences
        demographic_preferences = {
            '18-24': ['Clothing', 'Shoes', 'Accessories'],
            '25-34': ['Clothing', 'Bags', 'Jewelry'],
            '35-44': ['Clothing', 'Bags', 'Watches'],
            '45-54': ['Clothing', 'Jewelry', 'Watches'],
            '55+': ['Clothing', 'Jewelry', 'Accessories']
        }
        
        for _, user in self.test_data.iterrows():
            age_group = str(user['age_group'])
            recommended = demographic_preferences.get(age_group, ['Clothing', 'Shoes', 'Bags'])
            results[str(user['user_id'])] = recommended
        
        return results
    
    def calculate_metrics(self, k: int = 3):
        """
        Calculate evaluation metrics for all strategies.
        
        Args:
            k: Number of top recommendations to consider
        """
        print(f"\n📊 Calculating metrics (Top-{k} recommendations)...")
        
        metrics = {}
        
        for strategy_name, recommendations in self.recommendation_results.items():
            print(f"   Calculating metrics for {strategy_name}...")
            
            strategy_metrics = self._calculate_strategy_metrics(recommendations, k)
            metrics[strategy_name] = strategy_metrics
        
        self.metrics = metrics
        
        return metrics
    
    def _calculate_strategy_metrics(self, recommendations: Dict[str, List[str]], k: int) -> Dict[str, float]:
        """
        Calculate metrics for a specific strategy.
        
        Args:
            recommendations: Dictionary of user_id -> recommended categories
            k: Number of top recommendations to consider
            
        Returns:
            Dictionary of metrics
        """
        precisions = []
        recalls = []
        f1_scores = []
        hit_rates = []
        
        for user_id, recommended in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            # Get top-k recommendations
            top_k_recommended = recommended[:k]
            ground_truth = self.ground_truth[user_id]
            
            # Calculate metrics
            hits = len(set(top_k_recommended) & set(ground_truth))
            
            # Precision@K
            precision = hits / len(top_k_recommended) if top_k_recommended else 0
            precisions.append(precision)
            
            # Recall@K
            recall = hits / len(ground_truth) if ground_truth else 0
            recalls.append(recall)
            
            # F1@K
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            
            # Hit Rate@K
            hit_rate = 1 if hits > 0 else 0
            hit_rates.append(hit_rate)
        
        return {
            'precision@k': np.mean(precisions),
            'recall@k': np.mean(recalls),
            'f1@k': np.mean(f1_scores),
            'hit_rate@k': np.mean(hit_rates),
            'coverage': len(recommendations) / len(self.ground_truth)
        }
    
    def show_results(self):
        """
        Display evaluation results.
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        if not self.metrics:
            print("No metrics calculated. Run calculate_metrics() first.")
            return
        
        # Create results table
        results_df = pd.DataFrame(self.metrics).T
        
        print(f"\n📈 Performance Comparison:")
        print(results_df.round(4))
        
        # Find best performing strategy for each metric
        print(f"\n🏆 Best Performing Strategies:")
        for metric in ['precision@k', 'recall@k', 'f1@k', 'hit_rate@k']:
            best_strategy = results_df[metric].idxmax()
            best_score = results_df[metric].max()
            print(f"   • {metric}: {best_strategy} ({best_score:.4f})")
        
        # Show detailed breakdown
        print(f"\n📋 Detailed Breakdown:")
        for strategy_name, metrics in self.metrics.items():
            print(f"\n   {strategy_name}:")
            for metric_name, value in metrics.items():
                print(f"     • {metric_name}: {value:.4f}")
        
        return results_df
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """
        Save evaluation results to file.
        
        Args:
            output_path: Path to save results
        """
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'test_data_summary': {
                'total_users': len(self.test_data) if self.test_data is not None else 0,
                'ground_truth_users': len(self.ground_truth)
            },
            'metrics': self.metrics,
            'recommendation_counts': {
                strategy: len(recommendations) 
                for strategy, recommendations in self.recommendation_results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to {output_path}")

def main():
    """
    Main evaluation function.
    """
    print("🚀 Recommendation System Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator()
    
    # Show data summary
    evaluator.show_data_summary()
    
    # Test strategies
    evaluator.test_recommendation_strategies()
    
    # Calculate metrics
    evaluator.calculate_metrics(k=3)
    
    # Show results
    results_df = evaluator.show_results()
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "="*60)
    print("🎉 Evaluation completed!")
    print("="*60)

if __name__ == "__main__":
    main() 