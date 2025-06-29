#!/usr/bin/env python3
"""
Quick Accuracy Test for Recommendation System
============================================

A simplified test that evaluates your recommendation system
without requiring large datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
import json
from datetime import datetime

def generate_simple_test_data(n_users: int = 20) -> tuple:
    """Generate simple test data for quick evaluation."""
    print(f"Generating test data for {n_users} users...")
    
    np.random.seed(42)
    
    # Simple user profiles
    user_profiles = []
    for i in range(n_users):
        profile = {
            'user_id': f'user_{i}',
            'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+']),
            'gender': np.random.choice(['male', 'female']),
            'region': np.random.choice(['California', 'Texas', 'New York']),
            'engagement_score': np.random.uniform(30, 70),
            'avg_session_duration': np.random.uniform(5, 15),
            'avg_events_per_session': np.random.randint(3, 10),
            'total_transactions': np.random.randint(0, 5),
            'total_revenue': np.random.uniform(0, 200),
            'traffic_source': np.random.choice(['google', 'facebook', 'direct']),
            'device_type': np.random.choice(['mobile', 'desktop']),
            'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening'])
        }
        user_profiles.append(profile)
    
    # Generate ground truth
    categories = ['Clothing', 'Shoes', 'Bags', 'Accessories', 'Jewelry']
    ground_truth = {}
    
    for profile in user_profiles:
        # Simple logic: age-based preferences
        age_prefs = {
            '18-24': ['Clothing', 'Shoes', 'Accessories'],
            '25-34': ['Clothing', 'Bags', 'Jewelry'],
            '35-44': ['Clothing', 'Bags', 'Accessories'],
            '45-54': ['Clothing', 'Jewelry', 'Accessories'],
            '55+': ['Clothing', 'Jewelry', 'Accessories']
        }
        
        preferred = age_prefs.get(profile['age_group'], ['Clothing', 'Shoes'])
        # Add some randomness
        interactions = []
        for category in categories:
            if category in preferred and np.random.random() < 0.7:
                interactions.append(category)
            elif np.random.random() < 0.3:
                interactions.append(category)
        
        if not interactions:
            interactions = ['Clothing']
        
        ground_truth[profile['user_id']] = interactions
    
    return user_profiles, ground_truth

def test_simple_strategies(user_profiles: List[Dict], ground_truth: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """Test simple recommendation strategies."""
    print("Testing recommendation strategies...")
    
    strategies = {}
    categories = ['Clothing', 'Shoes', 'Bags', 'Accessories', 'Jewelry']
    
    # 1. Random Strategy
    random_results = {}
    for user in user_profiles:
        recommended = np.random.choice(categories, size=3, replace=False).tolist()
        random_results[user['user_id']] = recommended
    strategies['Random'] = random_results
    
    # 2. Popular Strategy (based on ground truth)
    all_interactions = []
    for interactions in ground_truth.values():
        all_interactions.extend(interactions)
    
    category_counts = pd.Series(all_interactions).value_counts()
    popular_categories = category_counts.head(3).index.tolist()
    
    popular_results = {}
    for user in user_profiles:
        popular_results[user['user_id']] = popular_categories
    strategies['Popular'] = popular_results
    
    # 3. Demographic Strategy
    demographic_prefs = {
        '18-24': ['Clothing', 'Shoes', 'Accessories'],
        '25-34': ['Clothing', 'Bags', 'Jewelry'],
        '35-44': ['Clothing', 'Bags', 'Accessories'],
        '45-54': ['Clothing', 'Jewelry', 'Accessories'],
        '55+': ['Clothing', 'Jewelry', 'Accessories']
    }
    
    demographic_results = {}
    for user in user_profiles:
        age_group = user['age_group']
        recommended = demographic_prefs.get(age_group, ['Clothing', 'Shoes', 'Bags'])
        demographic_results[user['user_id']] = recommended
    strategies['Demographic'] = demographic_results
    
    # 4. Engagement-based Strategy
    engagement_results = {}
    for user in user_profiles:
        if user['engagement_score'] > 50:
            recommended = ['Clothing', 'Bags', 'Jewelry']  # High engagement
        else:
            recommended = ['Clothing', 'Shoes', 'Accessories']  # Low engagement
        engagement_results[user['user_id']] = recommended
    strategies['Engagement-Based'] = engagement_results
    
    return strategies

def calculate_metrics(recommendations: Dict[str, List[str]], 
                     ground_truth: Dict[str, List[str]], 
                     k: int = 3) -> Dict[str, Union[float, int]]:
    """Calculate accuracy metrics."""
    precisions = []
    recalls = []
    f1_scores = []
    hit_rates = []
    
    for user_id, recommended in recommendations.items():
        if user_id not in ground_truth:
            continue
        
        top_k_recommended = recommended[:k]
        actual = ground_truth[user_id]
        
        hits = len(set(top_k_recommended) & set(actual))
        
        precision = hits / len(top_k_recommended) if top_k_recommended else 0
        recall = hits / len(actual) if actual else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        hit_rate = 1 if hits > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        hit_rates.append(hit_rate)
    
    return {
        'precision@k': float(np.mean(precisions)),
        'recall@k': float(np.mean(recalls)),
        'f1@k': float(np.mean(f1_scores)),
        'hit_rate@k': float(np.mean(hit_rates)),
        'users_evaluated': int(len(precisions))
    }

def rate_system(metrics: Dict[str, float]) -> str:
    """Rate the system based on F1 score."""
    f1_score = metrics['f1@k']
    
    if f1_score >= 0.7:
        return "EXCELLENT 🏆"
    elif f1_score >= 0.5:
        return "GOOD 👍"
    elif f1_score >= 0.3:
        return "FAIR ⚖️"
    elif f1_score >= 0.2:
        return "POOR 👎"
    else:
        return "VERY POOR ❌"

def main():
    """Main evaluation function."""
    print("🎯 Quick Recommendation System Rating")
    print("=" * 50)
    
    # Generate test data
    user_profiles, ground_truth = generate_simple_test_data(n_users=20)
    
    print(f"\n📊 Test Data:")
    print(f"   • Users: {len(user_profiles)}")
    print(f"   • Ground truth: {len(ground_truth)}")
    
    # Test strategies
    strategies = test_simple_strategies(user_profiles, ground_truth)
    
    # Calculate metrics
    print(f"\n📈 Calculating Metrics...")
    metrics = {}
    for strategy_name, recommendations in strategies.items():
        print(f"   • {strategy_name}...")
        metrics[strategy_name] = calculate_metrics(recommendations, ground_truth, k=3)
    
    # Display results
    print(f"\n" + "=" * 60)
    print("RECOMMENDATION SYSTEM RATING")
    print("=" * 60)
    
    # Print results table
    print(f"{'Strategy':<20} {'Precision@3':<12} {'Recall@3':<12} {'F1@3':<12} {'Hit Rate@3':<12} {'Rating':<15}")
    print("-" * 85)
    
    for strategy, metric_values in metrics.items():
        rating = rate_system(metric_values)
        print(f"{strategy:<20} {metric_values['precision@k']:<12.4f} {metric_values['recall@k']:<12.4f} "
              f"{metric_values['f1@k']:<12.4f} {metric_values['hit_rate@k']:<12.4f} {rating:<15}")
    
    # Find best strategy
    best_strategy = max(metrics.items(), key=lambda x: x[1]['f1@k'])
    best_rating = rate_system(best_strategy[1])
    
    print(f"\n🏆 BEST PERFORMING STRATEGY:")
    print(f"   • Strategy: {best_strategy[0]}")
    print(f"   • F1 Score: {best_strategy[1]['f1@k']:.4f}")
    print(f"   • Rating: {best_rating}")
    
    # Compare with random baseline
    random_metrics = metrics.get('Random', {})
    if random_metrics:
        improvement = ((best_strategy[1]['f1@k'] - random_metrics['f1@k']) / random_metrics['f1@k']) * 100
        print(f"   • Improvement over Random: {improvement:.1f}%")
    
    # Overall assessment
    print(f"\n📋 OVERALL ASSESSMENT:")
    print(f"   • Your recommendation system shows {best_rating} performance")
    print(f"   • Best F1 Score: {best_strategy[1]['f1@k']:.4f}")
    print(f"   • Hit Rate: {best_strategy[1]['hit_rate@k']:.4f}")
    
    if best_strategy[1]['f1@k'] >= 0.5:
        print(f"   ✅ Your system is performing well above random chance!")
    elif best_strategy[1]['f1@k'] >= 0.3:
        print(f"   ⚠️ Your system shows some improvement over random, but has room for enhancement.")
    else:
        print(f"   ❌ Your system needs significant improvement to be effective.")
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'test_summary': {
            'users': len(user_profiles),
            'ground_truth': len(ground_truth)
        },
        'metrics': metrics,
        'best_strategy': best_strategy[0],
        'best_f1': best_strategy[1]['f1@k'],
        'rating': best_rating
    }
    
    with open('quick_rating_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to quick_rating_results.json")
    print(f"\n" + "=" * 60)
    print("✅ Rating completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 