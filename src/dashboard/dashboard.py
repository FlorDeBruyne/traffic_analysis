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
        
        # Convert timestamp using custom parser
        df['time_stamp'] = df['time_stamp'].apply(parse_custom_timestamp)
        
        # Group by time period
        if 'daily' in collection_name.lower():
            df['time_stamp'] = df['time_stamp'].dt.floor('D')
        elif 'monthly' in collection_name.lower():
            df['time_stamp'] = df['time_stamp'].dt.floor('M')
        elif 'yearly' in collection_name.lower():
            df['time_stamp'] = df['time_stamp'].dt.floor('Y')
            
        # Aggregate data by timestamp and class_name
        df = df.groupby(['time_stamp', 'class_name']).agg({
            'detections_count': 'sum',
            'confidence_avg': 'mean',
            'confidence_max': 'max',
            'confidence_min': 'min'
        }).reset_index()
        
        # Sort by timestamp
        df = df.sort_values('time_stamp')
    
    return df

@st.cache_data
def load_evaluation_metrics():
    db = init_db()
    db.select_collection('evaluation_metrics')
    data = list(db.find({}))
    df = pd.DataFrame(data)
    
    if not df.empty:
        try:
            # Extract nested metrics safely
            metrics = ['accuracy', 'mean_ap', 'average_iou', 'precision', 'recall', 'f1']
            for metric in metrics:
                df[metric] = df['metrics'].apply(lambda x: 
                    x.get('overall', {}).get(metric, None) 
                    if isinstance(x, dict) and isinstance(x.get('overall'), dict) 
                    else None
                )
            
            # Convert timestamp using custom parser
            df['timestamp'] = df['timestamp'].apply(parse_custom_timestamp)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Remove rows where all metrics are None
            df = df.dropna(subset=metrics, how='all')
            
            if df.empty:
                st.warning("No valid evaluation metrics found after processing.")
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
col1, col2, col3 = st.columns(3)

with col1:
    if not category_df.empty:
        # Confidence metrics by class
        fig_conf = px.box(
            category_df,
            x='class_name',
            y='confidence_avg',
            title="Confidence Distribution by Class"
        )
        st.plotly_chart(fig_conf)
with col2:
    if not category_df.empty:
        # Distribution of trained vs untrained
        fig_dist = px.pie(
            names=['Trained', 'Untrained'],
            values=[category_df['trained_count'].sum(), category_df['untrained_count'].sum()],
            title="Training Distribution"
        )
        st.plotly_chart(fig_dist)
with col3:
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
            title="Average Processing Time by Stage"
        )
        st.plotly_chart(fig_speed)
# Evaluation Metrics Section
st.header("Model Evaluation Metrics")
eval_df = load_evaluation_metrics()

if not eval_df.empty:
    col1, col2, col3 = st.columns(3)

    with col1:
        # Precision-Recall curve
        if 'precision' in eval_df.columns and 'recall' in eval_df.columns:
            fig_pr = px.scatter(
                eval_df,
                x='recall',
                y='precision',
                title="Precision-Recall Curve"
            )
            st.plotly_chart(fig_pr)
        else:
            st.warning("Precision-Recall data not available")

    with col2:
        # Model metrics over time
        metrics = ['accuracy', 'mean_ap', 'f1']
        available_metrics = [m for m in metrics if m in eval_df.columns]
        if available_metrics:
            metrics_fig = px.line(
                eval_df,
                x='timestamp',
                y=available_metrics,
                title="Model Metrics Over Time"
            )
            st.plotly_chart(metrics_fig)
        else:
            st.warning("Time series metrics not available")

    with col3:
        # Latest metrics
        if len(eval_df) > 0:
            latest_eval = eval_df.iloc[-1]
            metrics_to_display = {
                'Accuracy': 'accuracy',
                'mAP': 'mean_ap',
                'F1': 'f1',
                'IoU': 'average_iou'
            }
            
            for label, metric in metrics_to_display.items():
                if metric in latest_eval and pd.notna(latest_eval[metric]):
                    st.metric(f"Latest {label}", f"{latest_eval[metric]:.3f}")
else:
    st.warning("No evaluation metrics available")

if st.checkbox("Show Debug Information"):
    st.write("Evaluation Metrics DataFrame Info:")
    st.write(eval_df.info())
    st.write("\nFirst few rows of evaluation metrics:")
    st.write(eval_df.head())
    
    st.write("\nTime Metrics DataFrame Info:")
    st.write(time_df.info())
    st.write("\nFirst few rows of time metrics:")
    st.write(time_df.head())


# Time Metrics Section
st.header(f"{time_period} Metrics")
col1, col2 = st.columns(2)

with col1:
    if not time_df.empty:
        # Detection count over time
        fig_det = px.line(
            time_df,
            x='time_stamp',
            y='detections_count',
            color='class_name',
            title=f"Detection Count Over Time"
        )
        st.plotly_chart(fig_det)

with col2:
    if not time_df.empty:
        # Confidence trends
        fig_conf_trend = px.line(
            time_df,
            x='time_stamp',
            y=['confidence_avg', 'confidence_max', 'confidence_min'],
            title="Confidence Metrics Over Time"
        )
        st.plotly_chart(fig_conf_trend)


