import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from src.db.mongo_instance import MongoInstance

# Initialize MongoDB connection
@st.cache_resource
def init_db():
    client = MongoInstance()
    return client

# Data loading functions
@st.cache_data
def load_category_metrics():
    db = init_db()
    db.select_collection('category_metrics')
    return list(db.find())

@st.cache_data
def load_time_metrics(collection_name):
    db = init_db()
    db.select_collection(collection_name)
    return list(db.find())

@st.cache_data
def load_evaluation_metrics():
    db = init_db()
    db.select_collection('evaluatation_metrics')
    return list(db.find())

# Set page config
st.set_page_config(page_title="YOLO Results Dashboard", layout="wide")

# Sidebar for filtering
st.sidebar.title("Filters")
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["Daily", "Monthly", "Yearly"]
)

# Main dashboard
st.title("YOLO Detection Results Dashboard")

# Load data
category_data = load_category_metrics()
time_data = load_time_metrics(f"{time_period.lower()}_metrics")
eval_data = load_evaluation_metrics()

# Category Metrics Section
st.header("Category Metrics")
col1, col2, col3 = st.columns(3)

# Convert category data to DataFrame
cat_df = pd.DataFrame(category_data)

with col1:
    # Distribution of trained vs untrained
    fig_dist = px.pie(
        names=['Trained', 'Untrained'],
        values=[cat_df['trained_count'].sum(), cat_df['untrained_count'].sum()],
        title="Training Distribution"
    )
    st.plotly_chart(fig_dist)

with col2:
    # Confidence metrics by class
    fig_conf = px.box(
        cat_df,
        x='class_name',
        y=['confidence_metrics.average'],
        title="Confidence Distribution by Class"
    )
    st.plotly_chart(fig_conf)

with col3:
    # Speed metrics overview
    speed_data = []
    for metric in ['preprocess', 'inference', 'postprocess']:
        speed_data.append({
            'stage': metric,
            'average_time': cat_df[f'speed_metrics.{metric}.avg'].mean()
        })
    
    fig_speed = px.bar(
        speed_data,
        x='stage',
        y='average_time',
        title="Average Processing Time by Stage"
    )
    st.plotly_chart(fig_speed)

# Time Metrics Section
st.header(f"{time_period} Metrics")
col1, col2 = st.columns(2)

# Convert time data to DataFrame
time_df = pd.DataFrame(time_data)
time_df['time_stamp'] = pd.to_datetime(time_df['time_stamp'])

with col1:
    # Detection count over time
    fig_det = px.line(
        time_df,
        x='time_stamp',
        y='detection_count',
        color='class_name',
        title=f"Detection Count Over Time"
    )
    st.plotly_chart(fig_det)

with col2:
    # Confidence trends
    fig_conf_trend = px.line(
        time_df,
        x='time_stamp',
        y=['confidence_metrics.avg', 'confidence_metrics.max', 'confidence_metrics.min'],
        title="Confidence Metrics Over Time"
    )
    st.plotly_chart(fig_conf_trend)

# Evaluation Metrics Section
st.header("Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)

# Convert evaluation data to DataFrame
eval_df = pd.DataFrame(eval_data)
eval_df['timestamp'] = pd.to_datetime(eval_df['timestamp'])

with col1:
    # Precision-Recall curve
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=eval_df['metrics.overall.recall'],
        y=eval_df['metrics.overall.precision'],
        mode='lines+markers'
    ))
    fig_pr.update_layout(title="Precision-Recall Curve")
    st.plotly_chart(fig_pr)

with col2:
    # Model metrics over time
    metrics_fig = px.line(
        eval_df,
        x='timestamp',
        y=['metrics.overall.accuracy', 'metrics.overall.mean_ap', 'metrics.overall.f1'],
        title="Model Metrics Over Time"
    )
    st.plotly_chart(metrics_fig)

with col3:
    # Latest metrics
    latest_eval = eval_df.iloc[-1]
    st.metric("Latest Accuracy", f"{latest_eval['metrics.overall.accuracy']:.3f}")
    st.metric("Latest mAP", f"{latest_eval['metrics.overall.mean_ap']:.3f}")
    st.metric("Latest F1", f"{latest_eval['metrics.overall.f1']:.3f}")
    st.metric("Latest IoU", f"{latest_eval['metrics.overall.average_iou']:.3f}")

# Add download buttons for the data
st.sidebar.markdown("### Download Data")
if st.sidebar.button("Download Category Metrics"):
    cat_df.to_csv("category_metrics.csv")
if st.sidebar.button("Download Time Metrics"):
    time_df.to_csv("time_metrics.csv")
if st.sidebar.button("Download Evaluation Metrics"):
    eval_df.to_csv("evaluation_metrics.csv")