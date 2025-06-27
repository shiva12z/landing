"""
advanced_recommendation_workflow.py

A full workflow for advanced recommendation logic using 'dataset1_final.csv' (user activity) and 'dataset 2.csv' (transactions).

Steps:
1. Data understanding & preprocessing (join, session construction, engagement categorization)
2. User segmentation (behavioral/demographic)
3. Cold start strategy (clustering, KNN, trends)
4. Personalization logic (rules/models for landing page content)
5. Evaluation and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# 1. Data Loading & Preprocessing
activity_df = pd.read_csv('dataset1_final.csv')
transaction_df = pd.read_csv('dataset2_final.csv')

# Merge datasets on user_id (adjust as needed)
if 'user_id' in activity_df.columns and 'user_id' in transaction_df.columns:
    df = pd.merge(activity_df, transaction_df, on='user_id', how='outer', suffixes=('_activity', '_transaction'))
else:
    raise ValueError('user_id column missing in one of the datasets')

# Session Construction (example: group by user and session_id if available)
if 'session_id' in df.columns:
    sessions = df.groupby(['user_id', 'session_id'])
else:
    # Fallback: treat each row as a session
    df['session_id'] = df.groupby('user_id').cumcount()
    sessions = df.groupby(['user_id', 'session_id'])

# Engagement Categorization (example: based on event_type or activity)
def categorize_engagement(row):
    if 'event_type' in row:
        if row['event_type'] in ['purchase', 'checkout']:
            return 'high'
        elif row['event_type'] in ['add_to_cart', 'wishlist']:
            return 'medium'
        else:
            return 'low'
    return 'unknown'

df['engagement_level'] = df.apply(categorize_engagement, axis=1)

# 2. User Segmentation (behavioral & demographic)
# Example: cluster users by total purchases and average session length
user_features = df.groupby('user_id').agg({
    'engagement_level': lambda x: Counter(x).most_common(1)[0][0],
    'product_id': 'nunique',
    'session_id': 'nunique',
    'amount': 'sum' if 'amount' in df.columns else lambda x: 0
}).reset_index()

# Encode categorical features
le = LabelEncoder()
user_features['engagement_level_enc'] = le.fit_transform(user_features['engagement_level'])

# Fill missing values
user_features = user_features.fillna(0)

# Clustering for segmentation
scaler = StandardScaler()
X = scaler.fit_transform(user_features[['product_id', 'session_id', 'amount', 'engagement_level_enc']])
kmeans = KMeans(n_clusters=3, random_state=42)
user_features['segment'] = kmeans.fit_predict(X)

# 3. Cold Start Strategy
# For new users: recommend top trending products in their segment
trending_products = df['product_id'].value_counts().index.tolist()
def recommend_for_cold_start(segment, top_n=5):
    # Recommend top products among users in the same segment
    seg_users = user_features[user_features['segment'] == segment]['user_id']
    seg_products = df[df['user_id'].isin(seg_users)]['product_id'].value_counts().index.tolist()
    return seg_products[:top_n] if seg_products else trending_products[:top_n]

# KNN for similar users (personalization)
knn = NearestNeighbors(n_neighbors=3)
knn.fit(X)

def recommend_for_user(user_id, top_n=5):
    if user_id not in user_features['user_id'].values:
        # Cold start
        segment = 0
        return recommend_for_cold_start(segment, top_n)
    idx = user_features[user_features['user_id'] == user_id].index[0]
    _, neighbors = knn.kneighbors([X[idx]])
    neighbor_ids = user_features.iloc[neighbors[0]]['user_id'].tolist()
    # Aggregate products from neighbors
    neighbor_products = df[df['user_id'].isin(neighbor_ids)]['product_id'].value_counts().index.tolist()
    return neighbor_products[:top_n] if neighbor_products else trending_products[:top_n]

# 4. Personalization Logic Example
# For each user, recommend products for landing page
user_recommendations = {}
for uid in user_features['user_id']:
    user_recommendations[uid] = recommend_for_user(uid, top_n=5)

# 5. Evaluation & Visualization
# Example: Precision@5 (if ground truth available)
def evaluate_precision_at_k(user_recs, ground_truth, k=5):
    precisions = []
    for uid, recs in user_recs.items():
        if uid in ground_truth:
            true_items = set(ground_truth[uid])
            hit = len(set(recs[:k]) & true_items)
            precisions.append(hit / k)
    return np.mean(precisions) if precisions else 0

# Example ground truth: last 5 products purchased by each user
user_gt = df.groupby('user_id')['product_id'].apply(lambda x: list(x)[-5:]).to_dict()
precision_at_5 = evaluate_precision_at_k(user_recommendations, user_gt, k=5)
print(f'Precision@5: {precision_at_5:.2f}')

# Visualization: User Segments
plt.figure(figsize=(8, 5))
plt.hist(user_features['segment'], bins=range(4), align='left', rwidth=0.7)
plt.xlabel('User Segment')
plt.ylabel('Number of Users')
plt.title('User Segmentation Distribution')
plt.xticks(range(3))
plt.show()

# Visualization: Engagement Level
plt.figure(figsize=(8, 5))
user_features['engagement_level'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Engagement Level')
plt.ylabel('Number of Users')
plt.title('User Engagement Level Distribution')
plt.show()

# Save recommendations to CSV
rec_df = pd.DataFrame([
    {'user_id': uid, 'recommendations': recs}
    for uid, recs in user_recommendations.items()
])
rec_df.to_csv('user_recommendations.csv', index=False)

print('Workflow complete. Recommendations saved to user_recommendations.csv.')
