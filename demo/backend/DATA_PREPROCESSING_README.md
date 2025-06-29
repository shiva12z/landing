# Data Preprocessing Workflow

This document describes the data preprocessing pipeline for joining activity and transaction datasets, constructing user sessions, and categorizing them based on engagement type.

## Overview

The data preprocessing workflow consists of the following main steps:

1. **Data Loading**: Load activity and transaction datasets
2. **Data Cleaning**: Handle missing values, convert data types, and remove invalid records
3. **Dataset Joining**: Join activity and transaction datasets using transaction_id field
4. **Session Construction**: Create meaningful user sessions based on time gaps
5. **Session Categorization**: Categorize sessions based on engagement type and behavior patterns
6. **Analysis**: Generate insights and summary statistics

## Files

- `data_preprocessing.py`: Main preprocessing pipeline class
- `session_analysis.py`: Session analysis and insights generation
- `demo_preprocessing.py`: Demo script with sample data processing
- `requirements.txt`: Python dependencies

## Dataset Structure

### Activity Dataset (dataset1_final.csv)
- **user_pseudo_id**: Unique user identifier
- **event_name**: Type of user event (session_start, page_view, etc.)
- **eventDate**: Date of the event
- **eventTimestamp**: Timestamp of the event
- **transaction_id**: Transaction identifier (if applicable)
- **page_type**: Type of page visited
- **category**: Event category
- **city, region, country**: Geographic information
- **source, medium**: Traffic source information
- **gender, Age**: User demographics
- **income_group**: User income level

### Transaction Dataset (dataset2_final.csv)
- **Date**: Transaction date
- **Transaction_ID**: Unique transaction identifier
- **Item_purchase_quantity**: Quantity of items purchased
- **Item_revenue**: Revenue from the transaction
- **ItemName**: Name of the purchased item
- **ItemBrand**: Brand of the item
- **ItemCategory**: Category of the item
- **ItemID**: Unique item identifier

## Session Categories

The preprocessing pipeline categorizes sessions into the following types:

1. **Purchase Session**: Sessions that resulted in a transaction
2. **High Engagement Session**: Sessions with ≥10 events or ≥15 minutes duration
3. **Product Browsing Session**: Sessions that included product page visits
4. **Cart Session**: Sessions that included cart page visits
5. **Checkout Session**: Sessions that included checkout page visits
6. **Quick Visit Session**: Sessions with ≤3 events and ≤2 minutes duration
7. **General Browsing Session**: All other sessions

## Engagement Levels

Sessions are also classified by engagement level:

- **High**: Purchase sessions or sessions with ≥8 events or ≥10 minutes duration
- **Medium**: Sessions with ≥4 events or ≥5 minutes duration
- **Low**: All other sessions

## Usage

### 1. Full Pipeline Processing

```python
from data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Run the complete pipeline
preprocessor.run_full_pipeline(session_timeout_minutes=30, save_data=True)
```

### 2. Step-by-Step Processing

```python
from data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Run individual steps
preprocessor.load_data()
preprocessor.clean_data()
preprocessor.join_datasets()
preprocessor.construct_sessions(session_timeout_minutes=30)
preprocessor.categorize_sessions()
preprocessor.generate_session_summary()
preprocessor.save_processed_data()
```

### 3. Session Analysis

```python
from session_analysis import SessionAnalyzer

# Initialize analyzer
analyzer = SessionAnalyzer()

# Load sessions and run analysis
if analyzer.load_sessions():
    analyzer.analyze_session_patterns()
    analyzer.analyze_user_behavior()
    analyzer.analyze_session_flow()
    analyzer.generate_insights()
    analyzer.save_analysis_report()
```

### 4. Demo Processing

```python
# Run demo with sample data
python demo_preprocessing.py
```

## Output Files

The preprocessing pipeline generates the following output files:

1. **merged_activity_transactions.csv**: Joined activity and transaction data
2. **user_sessions.csv**: Session-level aggregated data with categories
3. **session_analysis_report.txt**: Comprehensive analysis report

## Key Features

### Session Construction
- **Timeout-based**: Sessions are created based on 30-minute inactivity periods
- **User-specific**: Each user's sessions are tracked independently
- **Event-based**: Sessions include all events within the timeout period

### Session Categorization
- **Multi-criteria**: Uses event count, duration, page types, and transaction data
- **Hierarchical**: Purchase sessions take priority over other categories
- **Flexible**: Categories can be easily modified or extended

### Data Quality
- **Missing value handling**: Proper handling of missing transaction IDs and user IDs
- **Data type conversion**: Automatic conversion of dates and timestamps
- **Validation**: Removal of invalid records

## Performance Considerations

- **Memory efficient**: Processes data in chunks for large datasets
- **Scalable**: Can handle datasets with millions of records
- **Configurable**: Session timeout and categorization rules can be adjusted

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- python-dateutil: Date parsing utilities

## Installation

```bash
pip install -r requirements.txt
```

## Example Output

```
Starting Data Preprocessing Pipeline
==================================================
Loading datasets...
Loading activity data from dataset1_final.csv...
Activity dataset shape: (1000000, 15)
Loading transaction data from dataset2_final.csv...
Transaction dataset shape: (50000, 8)

Cleaning datasets...
Activity dataset shape after cleaning: (999500, 15)
Transaction dataset shape after cleaning: (50000, 8)

Joining datasets...
Merged dataset shape: (999500, 22)
Number of records with transactions: 45000
Number of records without transactions: 954500

Constructing user sessions (timeout: 30 minutes)...
Total sessions created: 150000
Average events per session: 6.67

Categorizing sessions...
Session Category Distribution:
General Browsing Session: 80000 sessions (53.3%)
Product Browsing Session: 40000 sessions (26.7%)
High Engagement Session: 20000 sessions (13.3%)
Purchase Session: 8000 sessions (5.3%)
Quick Visit Session: 2000 sessions (1.3%)

Engagement Level Distribution:
Low Engagement: 60000 sessions (40.0%)
Medium Engagement: 50000 sessions (33.3%)
High Engagement: 40000 sessions (26.7%)

Generating session summary...
Session Summary Statistics:
Total Sessions: 150000
Total Users: 75000
Sessions with Purchases: 8000
Average Session Duration (minutes): 8.45
Average Events per Session: 6.67
Total Revenue: $450000.00
Average Revenue per Purchase Session: $56.25

Data Preprocessing Pipeline Completed Successfully!
==================================================
```

## Troubleshooting

### Common Issues

1. **Memory Error**: For large datasets, consider processing in chunks
2. **File Not Found**: Ensure dataset files are in the correct directory
3. **Date Parsing Error**: Check date format in the datasets
4. **Missing Dependencies**: Install required packages using pip

### Performance Tips

1. Use SSD storage for faster I/O operations
2. Increase available RAM for large datasets
3. Consider using data sampling for initial testing
4. Adjust session timeout based on business requirements

## Future Enhancements

- Real-time session processing
- Advanced user segmentation
- Machine learning-based session categorization
- Integration with real-time analytics platforms
- Support for additional data sources 