import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.db.mongo_instance import MongoInstance


# Initialize MongoDB connection
@st.cache_resource
def init_db():
    client = MongoInstance('traffic_analysis')
    return client

def parse_custom_timestamp(timestamp_str):
    try:
        # Parse timestamp in format "YY_MM_DD_HH_MM_SS"
        parts = timestamp_str.split('_')
        if len(parts) == 6:
            year = int('20' + parts[0])  # Assuming 20xx year
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            second = int(parts[5])
            return pd.Timestamp(year, month, day, hour, minute, second)
    except Exception as e:
        st.error(f"Error parsing timestamp {timestamp_str}: {e}")
        return pd.NaT
    return pd.NaT

# Data loading functions
@st.cache_data
def load_category_metrics():
    db = init_db()
    db.select_collection('category_metrics')
    data = list(db.find({}))

    # Convert nested dictionaries to flat DataFrame columns
    df = pd.DataFrame(data)
    
    # Extract nested metrics
    if not df.empty:
        df['confidence_avg'] = df['confidence_metrics'].apply(lambda x: x['average'])
        df['confidence_max'] = df['confidence_metrics'].apply(lambda x: x['max'])
        df['confidence_min'] = df['confidence_metrics'].apply(lambda x: x['min'])
        
        # Extract speed metrics
        for stage in ['preprocess', 'inference', 'postprocess']:
            df[f'{stage}_avg'] = df['speed_metrics'].apply(lambda x: x[stage]['average'])
            df[f'{stage}_max'] = df['speed_metrics'].apply(lambda x: x[stage]['max'])
            df[f'{stage}_min'] = df['speed_metrics'].apply(lambda x: x[stage]['min'])
        
        # Extract box metrics
        df['box_width_avg'] = df['box_metrics'].apply(lambda x: x['width']['average'])
        df['box_height_avg'] = df['box_metrics'].apply(lambda x: x['height']['average'])
    
    return df

@st.cache_data
def load_time_metrics(collection_name):
    db = init_db()
    db.select_collection(collection_name)
    data = list(db.find({}))
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        # Extract nested metrics
        df['confidence_avg'] = df['confidence_metrics'].apply(lambda x: x['average'])
        df['confidence_max'] = df['confidence_metrics'].apply(lambda x: x['max'])
        df['confidence_min'] = df['confidence_metrics'].apply(lambda x: x['min'])
        
        # Handle different time periods using their explicit fields
        if 'daily' in collection_name.lower():
            # Extract year and month from timestamp for daily data
            df['year'] = df['time_stamp'].apply(lambda x: '20' + x.split('_')[0])
            df['month'] = df['time_stamp'].apply(lambda x: x.split('_')[1])
            # Use explicit day field
            df['day'] = df['day'].astype(str).str.zfill(2)
            
            # Create proper datetime for sorting
            df['sort_key'] = pd.to_datetime(
                df['year'] + '-' + 
                df['month'].str.zfill(2) + '-' + 
                df['day']
            )
            # Create display string (just the date)
            df['display_time'] = df['day']
            
        elif 'monthly' in collection_name.lower():
            # Use explicit month field
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            # Create sort key with year for proper sorting
            year = df['time_stamp'].apply(lambda x: '20' + x.split('_')[0])
            month = df['month'].astype(str).str.zfill(2)
            df['sort_key'] = pd.to_datetime(year + '-' + month + '-01')
            # Display string (just month name)
            df['display_time'] = df['month'].map(month_names)
            
        elif 'yearly' in collection_name.lower():
            # Convert to full year number (assuming 20xx)
            df['display_time'] = df['year'].apply(lambda x: f"20{x:02d}")
            # Create sort key
            df['sort_key'] = pd.to_datetime(df['display_time'] + '-01-01')
            
        # Aggregate data by timestamp and class_name
        df = df.groupby(['display_time', 'class_name']).agg({
            'detections_count': 'sum',
            'confidence_avg': 'mean',
            'confidence_max': 'max',
            'confidence_min': 'min',
            'sort_key': 'first'
        }).reset_index()
        
        # Sort by sort_key
        df = df.sort_values('sort_key')
    
    return df



@st.cache_data
def load_evaluation_metrics():
    db = init_db()
    db.select_collection('evaluation_metrics')
    data = list(db.find({}))
    df = pd.DataFrame(data)
    
    if not df.empty:
        try:
            metrics = ['accuracy', 'mean_ap', 'average_iou', 'precision', 'recall', 'f1']
            for metric in metrics:
                df[metric] = df['metrics'].apply(lambda x: 
                    float(x.get('overall', {}).get(metric, 0)) 
                    if isinstance(x, dict) and isinstance(x.get('overall'), dict) 
                    else 0.0
                )
            
            df['display_time'] = df['timestamp'].apply(parse_custom_timestamp)
            # df['display_time'] = df['timestamp'].dt.strftime('%Y-%m-%d')
            df = df.sort_values('display_time')
            df = df.dropna(subset=metrics, how='all')
            
        except Exception as e:
            st.error(f"Error processing evaluation metrics: {e}")
            return pd.DataFrame()
    
    return df

# Set page config
st.set_page_config(page_title="YOLO Results Dashboard", layout="wide")

# Load data
category_df = load_category_metrics()
time_period = st.sidebar.selectbox("Select Time Period", ["Daily", "Monthly", "Yearly"])
time_df = load_time_metrics(f"{time_period.lower()}_metrics")
eval_df = load_evaluation_metrics()

# Category Metrics Section
st.header("Category Metrics")
col1, col2 = st.columns(2)

with col1:
    if not category_df.empty:
        # Confidence metrics by class
        fig_conf = px.bar(
            category_df,
            x='class_name',
            y='confidence_avg',
            color='class_name',
            text_auto='.2f',
            title="Confidence Distribution by Class",
            text='class_name',
        )
        fig_conf.update_traces(textfont_size=30, textangle=0, textfont=dict(weight='bold'))
        st.plotly_chart(fig_conf)

with col2:
    if not category_df.empty:
        # Speed metrics overview
        speed_data = pd.DataFrame({
            'stage': ['Preprocess', 'Inference', 'Postprocess'],
            'average_time': [
                category_df['preprocess_avg'].mean(),
                category_df['inference_avg'].mean(),
                category_df['postprocess_avg'].mean()
            ]
        })
        fig_speed = px.bar(
            speed_data,
            x='stage',
            y='average_time',
            text='average_time',
            text_auto='.2f',
            color='stage',
            title="Average Processing Time by Stage"
        )
        fig_speed.update_traces(textfont_size=20, textangle=0, textfont=dict(weight='bold'))
        st.plotly_chart(fig_speed)

# Evaluation Metrics Section
st.header("Evaluation Metrics")
if not eval_df.empty:
    col1, col2, col3 = st.columns([1, 1, 1])  # Adjust column widths

    with col1:
        metrics = ['accuracy', 'mean_ap', 'precision', 'recall', 'f1']
        available_metrics = [m for m in metrics if m in eval_df.columns]
        if available_metrics:
            # Convert metric columns to numeric
            for metric in available_metrics:
                eval_df[metric] = pd.to_numeric(eval_df[metric], errors='coerce')

            metrics_fig = px.line(
                eval_df,
                y=available_metrics,
                title="Model Metrics Over Time"
            )
            metrics_fig.update_xaxes(title="Date")
            metrics_fig.update_layout(
                xaxis_tickangle=-45,
                yaxis_range=[0, 1]
            )
            st.plotly_chart(metrics_fig)

    with col2:
        # Split metrics into two groups
        st.subheader("Latest Metrics")
        latest_eval = eval_df.iloc[-1]
        metrics_to_display = {
            'Accuracy': 'accuracy',
            'mAP': 'mean_ap',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1': 'f1',
            'IoU': 'average_iou'
        }

        col2a, col2b = st.columns(2)  # Split column into two parts

        with col2a:
            for label, metric in list(metrics_to_display.items())[:3]:  # First 3 metrics
                if metric in latest_eval and pd.notna(latest_eval[metric]):
                    st.metric(f"Latest {label}", f"{latest_eval[metric]:.3f}")

        with col2b:
            for label, metric in list(metrics_to_display.items())[3:]:  # Remaining metrics
                if metric in latest_eval and pd.notna(latest_eval[metric]):
                    st.metric(f"Latest {label}", f"{latest_eval[metric]:.3f}")

    with col3:
        if 'precision' in eval_df.columns and 'recall' in eval_df.columns:
            fig_pr = px.scatter(
                eval_df,
                x='recall',
                y='precision',
                title="Precision-Recall Curve"
            )
            fig_pr.update_traces(mode='lines+markers')
            fig_pr.update_layout(
                xaxis_range=[0, 1],
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_pr)




# Time Metrics Section
st.header(f"{time_period} Metrics")
col1, col2 = st.columns(2)

with col1:
    if not time_df.empty:
        # Detection count over time
        fig_det = px.bar(
            time_df,
            x='display_time',
            y='detections_count',
            color='class_name',
            title=f"Detection Count Over Time"
        )
        fig_det.update_layout(
            bargap=0.2,
            xaxis_title=time_period.lower().rstrip('ly'),  # Convert "Daily" to "day", "Monthly" to "month", etc.
            xaxis={
                'categoryorder': 'array', 
                'categoryarray': sorted(time_df['display_time'].unique(), 
                                   key=lambda x: time_df.loc[time_df['display_time'] == x, 'sort_key'].iloc[0]),
                'dtick': 1,  # Force whole number intervals
                'type': 'category'  # Treat x-axis as categorical
            }
        )
        st.plotly_chart(fig_det)

with col2:
    if not time_df.empty:
        # Confidence trends
        fig_conf_trend = px.line(
            time_df,
            x='display_time',
            y=['confidence_avg', 'confidence_max', 'confidence_min'],
            title="Confidence Metrics Over Time"
        )
        fig_conf_trend.update_layout(
            xaxis_title=time_period.lower().rstrip('ly'),
            xaxis={
                'categoryorder': 'array', 
                'categoryarray': sorted(time_df['display_time'].unique(), 
                                   key=lambda x: time_df.loc[time_df['display_time'] == x, 'sort_key'].iloc[0]),
                'dtick': 1,  # Force whole number intervals
                'type': 'category'  # Treat x-axis as categorical
            },
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig_conf_trend)

# # Add debug information to help troubleshoot
# if st.checkbox("Show Debug Information about time metrics"):
#     st.write("Time Metrics DataFrame:")
#     st.write("Sort Keys:")
#     st.write(pd.DataFrame({
#         'Display Time': time_df['display_time'],
#         'Sort Key': time_df['sort_key']
#     }).drop_duplicates().sort_values('Sort Key'))
#     st.write("\nFull Time Metrics DataFrame:")
#     st.write(time_df)


# if st.checkbox("Show Debug Information"):
#     st.write("Evaluation Metrics DataFrame Info:")
#     st.write(eval_df.info(max_cols=25))
#     st.write("\nFirst few rows of evaluation metrics:")
#     st.write(eval_df.head())
    
#     st.write("\nTime Metrics DataFrame Info:")
#     st.write(time_df.info(max_cols=25))
#     st.write("\nFirst few rows of time metrics:")
#     st.write(time_df.head())