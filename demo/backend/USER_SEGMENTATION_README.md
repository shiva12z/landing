# User Segmentation Workflow

This document describes the user segmentation pipeline that groups users by behavioral and demographic attributes for targeted marketing and personalized experiences.

## Overview

The user segmentation workflow creates comprehensive user segments based on multiple dimensions:

1. **Engagement Levels**: Cart abandoners, frequent viewers, repeat purchasers
2. **Demographics**: Age groups, gender, income brackets
3. **Traffic Sources**: Paid vs organic traffic, social channels
4. **Geography**: City, state, country

## Files

- `user_segmentation.py`: Main segmentation pipeline class
- `demo_segmentation.py`: Demo script with sample data processing
- `USER_SEGMENTATION_README.md`: This documentation

## Segmentation Dimensions

### 1. Engagement Level Segments

**Categories:**
- **Repeat Purchaser**: Users with 2+ transactions
- **Single Purchaser**: Users with exactly 1 transaction
- **Cart Abandoner**: Users who visited cart but didn't purchase
- **High Engagement Viewer**: Users with long sessions (≥10 min) or many events (≥8)
- **Frequent Viewer**: Users with 3+ sessions
- **Occasional Viewer**: All other users

**Engagement Score (0-100):**
- Transaction weight: 40 points
- Session duration weight: 25 points
- Event count weight: 20 points
- Session frequency weight: 15 points

### 2. Demographic Segments

**Age Groups:**
- 18-24, 25-34, 35-44, 45-54, 55-64, 65+, Unknown

**Gender:**
- Male, Female, Unknown

**Income Brackets:**
- High Income (Top 10%)
- Above Average (11-20%)
- Average (21-50%)
- Low Income (Below 50%)
- Unknown

### 3. Traffic Source Segments

**Categories:**
- **Social Media**: Facebook, Instagram, Twitter, LinkedIn, TikTok, YouTube
- **Paid Traffic**: CPC, CPM, Paid Search, Paid Social
- **Organic Search**: Google, Organic traffic
- **Direct Traffic**: Direct visits, None medium
- **Email Marketing**: Email campaigns
- **Referral Traffic**: Referral links
- **Other**: All other sources

### 4. Geographic Segments

**US Regions:**
- Northeast, Midwest, South, West, Other

**Countries and Cities:**
- Cleaned geographic data with proper categorization

## Customer Segments

The system creates high-level customer segments combining all dimensions:

1. **High-Value Customer**: Revenue ≥$500 and 2+ transactions
2. **Loyal Customer**: Engagement score ≥70 and 1+ transaction
3. **High-Potential Customer**: Engagement score ≥60, no purchases
4. **At-Risk Customer**: 1+ transaction, engagement score ≤30
5. **New Customer**: 1 transaction, engagement score ≤50
6. **Casual Browser**: Engagement score ≤40, no purchases
7. **Regular Customer**: All other users

## Usage

### 1. Full Segmentation Pipeline

```python
from user_segmentation import UserSegmenter

# Initialize segmenter
segmenter = UserSegmenter()

# Run the complete pipeline
insights = segmenter.run_full_segmentation(save_data=True)
```

### 2. Step-by-Step Segmentation

```python
from user_segmentation import UserSegmenter

# Initialize segmenter
segmenter = UserSegmenter()

# Load data
segmenter.load_data()

# Create individual segments
engagement_segments = segmenter.segment_by_engagement_level()
demographic_segments = segmenter.segment_by_demographics()
traffic_segments = segmenter.segment_by_traffic_source()
geography_segments = segmenter.segment_by_geography()

# Create comprehensive segments
comprehensive_segments = segmenter.create_comprehensive_segments()

# Generate insights
insights = segmenter.generate_segment_insights()

# Save results
segmenter.save_segments()
```

### 3. Demo Processing

```python
# Run demo with sample data
python demo_segmentation.py
```

## Output Files

The segmentation pipeline generates the following output files:

1. **comprehensive_user_segments.csv**: Complete user segments with all attributes
2. **segment_insights.json**: Comprehensive insights and statistics
3. **engagement_segment_summary.csv**: Summary statistics by engagement segment
4. **customer_segment_summary.csv**: Summary statistics by customer segment

## Key Features

### Multi-Dimensional Segmentation
- Combines behavioral, demographic, and geographic data
- Creates actionable customer segments
- Provides engagement scoring system

### Comprehensive Analytics
- Segment distribution analysis
- Revenue analysis by segment
- Engagement pattern insights
- Geographic distribution

### Marketing-Ready Outputs
- Segment priority scoring
- Actionable insights
- Export-ready data formats
- Detailed segment summaries

## Example Output

```
USER SEGMENTATION INSIGHTS
============================================================

1. CUSTOMER SEGMENTS:
   High-Value Customer: 1,250 users (2.5%)
   Loyal Customer: 5,000 users (10.0%)
   High-Potential Customer: 7,500 users (15.0%)
   At-Risk Customer: 2,500 users (5.0%)
   New Customer: 8,750 users (17.5%)
   Regular Customer: 15,000 users (30.0%)
   Casual Browser: 10,000 users (20.0%)

2. ENGAGEMENT SEGMENTS:
   Occasional Viewer: 20,000 users (40.0%)
   Frequent Viewer: 12,500 users (25.0%)
   High Engagement Viewer: 8,750 users (17.5%)
   Cart Abandoner: 5,000 users (10.0%)
   Single Purchaser: 2,500 users (5.0%)
   Repeat Purchaser: 1,250 users (2.5%)

3. DEMOGRAPHICS:
   Dominant Age Group: 25-34
   Dominant Gender: female
   Dominant Income: Average

4. TRAFFIC SOURCES:
   Top Traffic Category: Social Media
   Top Primary Source: Facebook

5. REVENUE ANALYSIS:
   Total Revenue: $1,250,000.00
   Average Revenue per User: $25.00
   Top Revenue Segment: High-Value Customer
```

## Business Applications

### Marketing Campaigns
- Target high-potential customers with conversion campaigns
- Re-engage at-risk customers with retention campaigns
- Reward loyal customers with exclusive offers
- Nurture casual browsers with educational content

### Product Development
- Identify user needs by segment
- Prioritize features based on high-value customer feedback
- Optimize user experience for different engagement levels

### Customer Support
- Proactive outreach to at-risk customers
- Personalized support based on customer segment
- Resource allocation based on customer value

### Revenue Optimization
- Focus on high-value customer retention
- Convert high-potential customers
- Optimize pricing for different segments

## Performance Considerations

- **Scalable**: Handles large user datasets efficiently
- **Flexible**: Easy to modify segmentation rules
- **Extensible**: Can add new segmentation dimensions
- **Maintainable**: Clear code structure and documentation

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- json: Data serialization

## Installation

```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure data preprocessing is completed first
2. **Memory Issues**: Process data in chunks for large datasets
3. **Missing Data**: Handle missing values appropriately in segmentation logic

### Performance Tips

1. Use SSD storage for faster I/O operations
2. Increase available RAM for large datasets
3. Consider using data sampling for initial testing
4. Optimize segmentation rules based on business requirements

## Future Enhancements

- Machine learning-based segmentation
- Real-time segmentation updates
- Advanced behavioral modeling
- Integration with marketing automation platforms
- Predictive analytics for customer lifetime value 