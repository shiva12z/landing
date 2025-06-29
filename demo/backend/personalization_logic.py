#!/usr/bin/env python3
"""
Personalization Logic Development
================================

This module defines rules and models for what content to show on landing pages:
- Hero banners based on inferred interest or top-trending categories
- Product modules filtered by category popularity, age cohort interest
- CTA modules adapted to stage of user (e.g., discover, explore, buy now)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UserStage(Enum):
    """User journey stages for CTA personalization"""
    DISCOVER = "discover"
    EXPLORE = "explore"
    CONSIDER = "consider"
    BUY = "buy"
    RETURN = "return"

class ContentType(Enum):
    """Types of content that can be personalized"""
    HERO_BANNER = "hero_banner"
    PRODUCT_MODULE = "product_module"
    CTA_MODULE = "cta_module"
    CATEGORY_GRID = "category_grid"
    RECOMMENDATION_SECTION = "recommendation_section"

@dataclass
class UserProfile:
    """User profile with personalization attributes"""
    user_id: str
    age_group: str
    gender: str
    region: str
    traffic_source: str
    device_type: str
    time_of_day: str
    engagement_score: float
    user_stage: UserStage
    inferred_interests: List[str]
    category_preferences: Dict[str, float]
    session_count: int
    total_revenue: float
    first_name: Optional[str] = None
    last_visit_date: Optional[datetime] = None

@dataclass
class ContentRule:
    """Rule for content personalization"""
    rule_id: str
    content_type: ContentType
    conditions: Dict[str, Any]
    priority: int
    content_data: Dict[str, Any]
    fallback_content: Optional[Dict[str, Any]] = None

class PersonalizationEngine:
    """
    Main personalization engine that applies rules and models
    to determine what content to show to users.
    """
    
    def __init__(self, trends_data: Optional[Dict] = None):
        """
        Initialize the personalization engine.
        
        Args:
            trends_data: Pre-computed trends data from cold start strategy
        """
        self.trends_data = trends_data or {}
        self.content_rules = []
        self.category_weights = {}
        self.age_cohort_interests = {}
        self.device_preferences = {}
        self.time_based_patterns = {}
        
        # Initialize default rules
        self._initialize_default_rules()
        self._load_trends_data()
    
    def _initialize_default_rules(self):
        """Initialize default personalization rules"""
        
        # Hero Banner Rules
        self.add_rule(ContentRule(
            rule_id="hero_high_engagement",
            content_type=ContentType.HERO_BANNER,
            conditions={
                "engagement_score": {"min": 70, "max": 100},
                "user_stage": [UserStage.EXPLORE, UserStage.CONSIDER, UserStage.BUY]
            },
            priority=1,
            content_data={
                "banner_type": "featured_products",
                "title": "Discover Your Perfect Style",
                "subtitle": "Curated just for you",
                "cta_text": "Shop Now",
                "cta_link": "/personalized-collection"
            }
        ))
        
        self.add_rule(ContentRule(
            rule_id="hero_new_user",
            content_type=ContentType.HERO_BANNER,
            conditions={
                "session_count": {"max": 1},
                "user_stage": UserStage.DISCOVER
            },
            priority=2,
            content_data={
                "banner_type": "welcome",
                "title": "Welcome to Our Store",
                "subtitle": "Discover amazing products tailored to your style",
                "cta_text": "Start Exploring",
                "cta_link": "/categories"
            }
        ))
        
        # Product Module Rules
        self.add_rule(ContentRule(
            rule_id="products_by_category",
            content_type=ContentType.PRODUCT_MODULE,
            conditions={
                "inferred_interests": {"required": True},
                "category_preferences": {"min_count": 1}
            },
            priority=1,
            content_data={
                "module_type": "category_based",
                "title": "Recommended for You",
                "product_count": 8,
                "sort_by": "relevance"
            }
        ))
        
        self.add_rule(ContentRule(
            rule_id="products_trending",
            content_type=ContentType.PRODUCT_MODULE,
            conditions={
                "engagement_score": {"max": 50},
                "user_stage": UserStage.DISCOVER
            },
            priority=2,
            content_data={
                "module_type": "trending",
                "title": "Trending Now",
                "product_count": 6,
                "sort_by": "popularity"
            }
        ))
        
        # CTA Module Rules
        self.add_rule(ContentRule(
            rule_id="cta_discover",
            content_type=ContentType.CTA_MODULE,
            conditions={
                "user_stage": UserStage.DISCOVER,
                "session_count": {"max": 3}
            },
            priority=1,
            content_data={
                "cta_type": "explore",
                "title": "Find Your Style",
                "description": "Browse our curated collections",
                "button_text": "Explore Categories",
                "button_link": "/categories",
                "secondary_text": "Take our style quiz",
                "secondary_link": "/style-quiz"
            }
        ))
        
        self.add_rule(ContentRule(
            rule_id="cta_buy",
            content_type=ContentType.CTA_MODULE,
            conditions={
                "user_stage": UserStage.BUY,
                "engagement_score": {"min": 60}
            },
            priority=1,
            content_data={
                "cta_type": "purchase",
                "title": "Ready to Shop?",
                "description": "Complete your purchase with exclusive offers",
                "button_text": "Shop Now",
                "button_link": "/checkout",
                "secondary_text": "View cart",
                "secondary_link": "/cart"
            }
        ))
    
    def _load_trends_data(self):
        """Load and process trends data for personalization"""
        if not self.trends_data:
            return
        
        # Extract category weights from trends
        if 'regional' in self.trends_data:
            for region, data in self.trends_data['regional'].items():
                if 'top_categories' in data:
                    for cat_data in data['top_categories']:
                        category = cat_data.get('category', 'Unknown')
                        count = cat_data.get('session_count', 0)
                        if category not in self.category_weights:
                            self.category_weights[category] = {}
                        self.category_weights[category][region] = count
        
        # Extract time-based patterns
        if 'time_of_day' in self.trends_data:
            for time_period, data in self.trends_data['time_of_day'].items():
                self.time_based_patterns[time_period] = {
                    'top_categories': [cat['category'] for cat in data.get('top_categories', [])],
                    'engagement_level': self._get_engagement_level_for_time(time_period)
                }
        
        # Extract source-based preferences
        if 'source' in self.trends_data:
            for source, data in self.trends_data['source'].items():
                self.device_preferences[source] = {
                    'top_categories': [cat['category'] for cat in data.get('top_categories', [])],
                    'preferred_content_type': self._get_content_type_for_source(source)
                }
    
    def _get_engagement_level_for_time(self, time_period: str) -> str:
        """Determine engagement level based on time of day"""
        engagement_map = {
            'Morning': 'Medium',
            'Afternoon': 'High',
            'Evening': 'High',
            'Night': 'Low'
        }
        return engagement_map.get(time_period, 'Medium')
    
    def _get_content_type_for_source(self, source: str) -> str:
        """Determine preferred content type based on traffic source"""
        content_map = {
            'google': 'search_results',
            'facebook': 'social_feed',
            'instagram': 'visual_gallery',
            'direct': 'featured_products',
            'email': 'promotional'
        }
        return content_map.get(source, 'general')
    
    def add_rule(self, rule: ContentRule):
        """Add a new personalization rule"""
        self.content_rules.append(rule)
        # Sort rules by priority (higher priority first)
        self.content_rules.sort(key=lambda x: x.priority, reverse=True)
    
    def create_user_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """
        Create a user profile from raw user data.
        
        Args:
            user_data: Raw user data dictionary
            
        Returns:
            UserProfile: Processed user profile
        """
        # Determine user stage based on behavior
        user_stage = self._determine_user_stage(user_data)
        
        # Infer interests from available data
        inferred_interests = self._infer_user_interests(user_data)
        
        # Calculate category preferences
        category_preferences = self._calculate_category_preferences(user_data, inferred_interests)
        
        return UserProfile(
            user_id=user_data.get('user_id', 'unknown'),
            age_group=user_data.get('age_group', 'Unknown'),
            gender=user_data.get('gender', 'unknown'),
            region=user_data.get('region', 'Unknown'),
            traffic_source=user_data.get('traffic_source', 'unknown'),
            device_type=user_data.get('device_type', 'desktop'),
            time_of_day=user_data.get('time_of_day', 'Afternoon'),
            engagement_score=user_data.get('engagement_score', 50.0),
            user_stage=user_stage,
            inferred_interests=inferred_interests,
            category_preferences=category_preferences,
            session_count=user_data.get('session_count', 0),
            total_revenue=user_data.get('total_revenue', 0.0),
            first_name=user_data.get('first_name'),
            last_visit_date=user_data.get('last_visit_date')
        )
    
    def _determine_user_stage(self, user_data: Dict[str, Any]) -> UserStage:
        """Determine user's stage in the journey"""
        session_count = user_data.get('session_count', 0)
        engagement_score = user_data.get('engagement_score', 50.0)
        total_revenue = user_data.get('total_revenue', 0.0)
        
        if total_revenue > 0:
            return UserStage.RETURN if session_count > 5 else UserStage.BUY
        elif engagement_score > 70 and session_count > 3:
            return UserStage.CONSIDER
        elif engagement_score > 50 or session_count > 1:
            return UserStage.EXPLORE
        else:
            return UserStage.DISCOVER
    
    def _infer_user_interests(self, user_data: Dict[str, Any]) -> List[str]:
        """Infer user interests from available data"""
        interests = []
        
        # Age-based interests
        age_group = user_data.get('age_group', 'Unknown')
        age_interests = {
            '18-24': ['Clothing', 'Shoes', 'Accessories'],
            '25-34': ['Clothing', 'Bags', 'Jewelry'],
            '35-44': ['Clothing', 'Bags', 'Watches'],
            '45-54': ['Clothing', 'Jewelry', 'Watches'],
            '55+': ['Clothing', 'Jewelry', 'Accessories']
        }
        interests.extend(age_interests.get(age_group, ['Clothing']))
        
        # Region-based interests
        region = user_data.get('region', 'Unknown')
        if region in self.category_weights:
            top_categories = sorted(
                self.category_weights.items(),
                key=lambda x: x[1].get(region, 0),
                reverse=True
            )[:3]
            interests.extend([cat for cat, _ in top_categories])
        
        # Time-based interests
        time_of_day = user_data.get('time_of_day', 'Afternoon')
        if time_of_day in self.time_based_patterns:
            interests.extend(self.time_based_patterns[time_of_day]['top_categories'][:2])
        
        # Remove duplicates and return
        return list(set(interests))
    
    def _calculate_category_preferences(self, user_data: Dict[str, Any], 
                                      inferred_interests: List[str]) -> Dict[str, float]:
        """Calculate category preference scores"""
        preferences = {}
        
        # Base preferences from inferred interests
        for interest in inferred_interests:
            preferences[interest] = 0.8
        
        # Adjust based on region
        region = user_data.get('region', 'Unknown')
        if region in self.category_weights:
            for category, region_weights in self.category_weights.items():
                if region in region_weights:
                    max_weight = max(region_weights.values()) if region_weights.values() else 1
                    preferences[category] = region_weights[region] / max_weight
        
        # Adjust based on time of day
        time_of_day = user_data.get('time_of_day', 'Afternoon')
        if time_of_day in self.time_based_patterns:
            for category in self.time_based_patterns[time_of_day]['top_categories']:
                if category in preferences:
                    preferences[category] *= 1.2
                else:
                    preferences[category] = 0.6
        
        return preferences
    
    def get_personalized_content(self, user_profile: UserProfile, 
                               content_type: ContentType) -> Dict[str, Any]:
        """
        Get personalized content for a specific content type.
        
        Args:
            user_profile: User profile
            content_type: Type of content to personalize
            
        Returns:
            Dict: Personalized content data
        """
        # Find matching rules
        matching_rules = []
        for rule in self.content_rules:
            if rule.content_type == content_type and self._rule_matches(rule, user_profile):
                matching_rules.append(rule)
        
        if not matching_rules:
            # Return fallback content
            return self._get_fallback_content(content_type, user_profile)
        
        # Return highest priority rule content
        best_rule = matching_rules[0]
        content = best_rule.content_data.copy()
        
        # Enhance content with user-specific data
        content = self._enhance_content_with_profile(content, user_profile)
        
        return content
    
    def _rule_matches(self, rule: ContentRule, user_profile: UserProfile) -> bool:
        """Check if a rule matches the user profile"""
        for condition_key, condition_value in rule.conditions.items():
            if not self._condition_matches(condition_key, condition_value, user_profile):
                return False
        return True
    
    def _condition_matches(self, condition_key: str, condition_value: Any, 
                          user_profile: UserProfile) -> bool:
        """Check if a specific condition matches the user profile"""
        
        # Get user value
        user_value = getattr(user_profile, condition_key, None)
        
        if condition_key == "user_stage":
            return user_value in condition_value if isinstance(condition_value, list) else user_value == condition_value
        
        elif condition_key == "inferred_interests":
            return bool(user_value) if condition_value.get("required", False) else True
        
        elif condition_key == "category_preferences":
            min_count = condition_value.get("min_count", 0)
            return len(user_value) >= min_count if user_value is not None else False
        
        elif isinstance(condition_value, dict):
            # Range condition (min/max)
            if "min" in condition_value and user_value < condition_value["min"]:
                return False
            if "max" in condition_value and user_value > condition_value["max"]:
                return False
            return True
        
        else:
            # Direct comparison
            return user_value == condition_value
    
    def _enhance_content_with_profile(self, content: Dict[str, Any], 
                                    user_profile: UserProfile) -> Dict[str, Any]:
        """Enhance content with user-specific data"""
        enhanced_content = content.copy()
        
        # Add personalized recommendations
        if "product_count" in enhanced_content:
            recommended_categories = self._get_recommended_categories(user_profile)
            enhanced_content["recommended_categories"] = recommended_categories[:enhanced_content["product_count"]]
        
        # Add personalized messaging
        if "title" in enhanced_content:
            enhanced_content["title"] = self._personalize_message(
                enhanced_content["title"], user_profile
            )
        
        # Add user-specific CTAs
        if "cta_text" in enhanced_content:
            enhanced_content["cta_text"] = self._get_personalized_cta(user_profile)
        
        return enhanced_content
    
    def _get_recommended_categories(self, user_profile: UserProfile) -> List[str]:
        """Get recommended categories for the user"""
        # Sort categories by user preferences
        sorted_categories = sorted(
            user_profile.category_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [category for category, score in sorted_categories if score > 0.3]
    
    def _personalize_message(self, message: str, user_profile: UserProfile) -> str:
        """Personalize a message based on user profile"""
        personalized = message
        
        # Add name if available
        if hasattr(user_profile, 'first_name') and user_profile.first_name:
            personalized = f"Hi {user_profile.first_name}, {personalized}"
        
        # Add region-specific content
        if user_profile.region != 'Unknown':
            personalized = f"{personalized} in {user_profile.region}"
        
        return personalized
    
    def _get_personalized_cta(self, user_profile: UserProfile) -> str:
        """Get personalized CTA text based on user stage"""
        cta_map = {
            UserStage.DISCOVER: "Start Exploring",
            UserStage.EXPLORE: "Discover More",
            UserStage.CONSIDER: "Shop Now",
            UserStage.BUY: "Complete Purchase",
            UserStage.RETURN: "Welcome Back"
        }
        return cta_map.get(user_profile.user_stage, "Shop Now")
    
    def _get_fallback_content(self, content_type: ContentType, 
                            user_profile: UserProfile) -> Dict[str, Any]:
        """Get fallback content when no rules match"""
        fallback_content = {
            ContentType.HERO_BANNER: {
                "banner_type": "general",
                "title": "Welcome to Our Store",
                "subtitle": "Discover amazing products",
                "cta_text": "Shop Now",
                "cta_link": "/categories"
            },
            ContentType.PRODUCT_MODULE: {
                "module_type": "featured",
                "title": "Featured Products",
                "product_count": 6,
                "sort_by": "popularity"
            },
            ContentType.CTA_MODULE: {
                "cta_type": "general",
                "title": "Ready to Shop?",
                "description": "Explore our collections",
                "button_text": "Browse Categories",
                "button_link": "/categories"
            }
        }
        
        return fallback_content.get(content_type, {})
    
    def get_full_landing_page_content(self, user_profile: UserProfile) -> Dict[str, Any]:
        """
        Get complete personalized landing page content.
        
        Args:
            user_profile: User profile
            
        Returns:
            Dict: Complete landing page content structure
        """
        landing_page = {
            "user_profile": {
                "user_id": user_profile.user_id,
                "user_stage": user_profile.user_stage.value,
                "engagement_score": user_profile.engagement_score,
                "inferred_interests": user_profile.inferred_interests
            },
            "sections": {}
        }
        
        # Get personalized content for each section
        content_types = [ContentType.HERO_BANNER, ContentType.PRODUCT_MODULE, 
                        ContentType.CTA_MODULE, ContentType.CATEGORY_GRID]
        
        for content_type in content_types:
            section_name = content_type.value.replace('_', ' ')
            landing_page["sections"][section_name] = self.get_personalized_content(
                user_profile, content_type
            )
        
        # Add metadata
        landing_page["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "personalization_version": "1.0",
            "rules_applied": len(self.content_rules)
        }
        
        return landing_page
    
    def save_personalization_data(self, output_dir: str = "personalization_data"):
        """Save personalization rules and data"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save rules
        rules_data = []
        for rule in self.content_rules:
            # Convert enum values to strings for JSON serialization
            conditions = {}
            for key, value in rule.conditions.items():
                if isinstance(value, list):
                    conditions[key] = [v.value if hasattr(v, 'value') else v for v in value]
                elif hasattr(value, 'value'):
                    conditions[key] = value.value
                else:
                    conditions[key] = value
            
            rule_dict = {
                "rule_id": rule.rule_id,
                "content_type": rule.content_type.value,
                "conditions": conditions,
                "priority": rule.priority,
                "content_data": rule.content_data
            }
            rules_data.append(rule_dict)
        
        with open(f"{output_dir}/personalization_rules.json", "w") as f:
            json.dump(rules_data, f, indent=2)
        
        # Save trends data
        with open(f"{output_dir}/personalization_trends.json", "w") as f:
            json.dump(self.trends_data, f, indent=2)
        
        print(f"Personalization data saved to {output_dir}/")

# Example usage and testing
def create_sample_user_data() -> Dict[str, Any]:
    """Create sample user data for testing"""
    return {
        "user_id": "user_12345",
        "age_group": "25-34",
        "gender": "female",
        "region": "California",
        "traffic_source": "facebook",
        "device_type": "mobile",
        "time_of_day": "Evening",
        "engagement_score": 75.0,
        "session_count": 3,
        "total_revenue": 0.0,
        "last_visit_date": datetime.now() - timedelta(days=2)
    }

def demo_personalization():
    """Demonstrate the personalization engine"""
    print("🚀 Personalization Logic Demo")
    print("=" * 50)
    
    # Create personalization engine
    engine = PersonalizationEngine()
    
    # Create sample user profile
    user_data = create_sample_user_data()
    user_profile = engine.create_user_profile(user_data)
    
    print(f"\n👤 User Profile Created:")
    print(f"   • User ID: {user_profile.user_id}")
    print(f"   • Stage: {user_profile.user_stage.value}")
    print(f"   • Engagement: {user_profile.engagement_score}")
    print(f"   • Interests: {user_profile.inferred_interests}")
    
    # Get personalized content for different sections
    print(f"\n📱 Personalized Content:")
    
    content_types = [ContentType.HERO_BANNER, ContentType.PRODUCT_MODULE, ContentType.CTA_MODULE]
    
    for content_type in content_types:
        content = engine.get_personalized_content(user_profile, content_type)
        print(f"\n   {content_type.value.replace('_', ' ').title()}:")
        for key, value in content.items():
            if isinstance(value, list):
                print(f"     • {key}: {value[:3]}...")
            else:
                print(f"     • {key}: {value}")
    
    # Get full landing page content
    landing_page = engine.get_full_landing_page_content(user_profile)
    
    print(f"\n🎯 Complete Landing Page Structure:")
    print(f"   • Sections: {list(landing_page['sections'].keys())}")
    print(f"   • Rules Applied: {landing_page['metadata']['rules_applied']}")
    
    # Save personalization data
    engine.save_personalization_data()
    
    return engine, user_profile, landing_page

def evaluate_recommendations(recommendations, ground_truth, k=3):
    """
    recommendations: dict of user_id -> list of recommended items
    ground_truth: dict of user_id -> list of actual items
    k: number of top recommendations to consider
    """
    precisions, recalls, f1s, hit_rates = [], [], [], []

    for user_id in recommendations:
        recs = recommendations[user_id][:k]
        truth = ground_truth.get(user_id, [])
        if not truth:
            continue

        hits = len(set(recs) & set(truth))
        precision = hits / len(recs) if recs else 0
        recall = hits / len(truth) if truth else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        hit_rate = 1 if hits > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        hit_rates.append(hit_rate)

    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}:    {np.mean(recalls):.4f}")
    print(f"F1@{k}:        {np.mean(f1s):.4f}")
    print(f"Hit Rate@{k}:  {np.mean(hit_rates):.4f}")

# Example usage:
# recommendations = {'user1': ['Shoes', 'Bags', 'Clothing'], ...}
# ground_truth = {'user1': ['Shoes', 'Jewelry'], ...}
# evaluate_recommendations(recommendations, ground_truth, k=3)

if __name__ == "__main__":
    demo_personalization() 