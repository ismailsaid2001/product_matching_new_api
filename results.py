import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load your data
df_detailed = pd.read_csv('evaluation_detailed_20260109_105907.csv')
df_summary = pd.read_csv('evaluation_summary_20260109_105907.csv')

# 1. Dashboard Overview - 4 KPIs en haut
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Performance Overview', 'Decision Node Distribution', 
                   'Accuracy by Category', 'Cost Analysis', 
                   'Processing Time by Node', 'Language Performance'),
    specs=[[{"type": "indicator"}, {"type": "pie"}],
           [{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

# KPI Cards
fig.add_trace(go.Indicator(
    mode = "gauge+number+delta",
    value = 13.4,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Accuracy %"},
    gauge = {'axis': {'range': [None, 100]},
             'bar': {'color': "red" if 13.4 < 50 else "yellow" if 13.4 < 80 else "green"},
             'steps': [{'range': [0, 50], 'color': "lightgray"},
                      {'range': [50, 80], 'color': "gray"}],
             'threshold': {'line': {'color': "red", 'width': 4},
                          'thickness': 0.75, 'value': 90}}
), row=1, col=1)

# Decision Node Distribution
node_data = [77, 43, 22, 1]
node_labels = ['Database (53.8%)', 'LLM (30.1%)', 'T5 (15.4%)', 'Error (0.7%)']
fig.add_trace(go.Pie(
    labels=node_labels, 
    values=node_data,
    hole=0.4
), row=1, col=2)

# 2. Accuracy by Category
categories = ['Food', 'Equipment', 'Cleaning', 'Beverages']
accuracy_rates = [22.45, 25.0, 15.0, 0.0]
fig.add_trace(go.Bar(
    x=categories,
    y=accuracy_rates,
    marker_color=['green' if x > 20 else 'orange' if x > 10 else 'red' for x in accuracy_rates]
), row=2, col=1)

# 3. Cost Analysis
nodes = ['Database', 'T5', 'LLM']
costs = [0, 0, 0.067168]
fig.add_trace(go.Bar(
    x=nodes,
    y=costs,
    marker_color=['green', 'green', 'red']
), row=2, col=2)

# 4. Processing Time by Node
processing_times = [137.25, 2672.81, 11263.29]
fig.add_trace(go.Bar(
    x=nodes,
    y=processing_times,
    marker_color=['green', 'orange', 'red']
), row=3, col=1)

# 5. Language Performance
languages = ['French', 'Mixed', 'English', 'Spanish', 'Italian']
lang_accuracy = [11/98*100, 5/25*100, 1/5*100, 1/8*100, 1/6*100]  # Approximation
fig.add_trace(go.Bar(
    x=languages,
    y=lang_accuracy
), row=3, col=2)

fig.update_layout(
    title_text="🚀 Product Classification System - Performance Dashboard",
    showlegend=False,
    height=1000
)

fig.show()
fig.write_html("classification_dashboard.html")