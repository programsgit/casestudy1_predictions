import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np


# ===== Page Configuration =====
st.set_page_config(page_title="Clustering of Mall Customers", page_icon=":shopping_bag:", layout="wide")
st.title(":people_holding_hands::shopping_trolley: Mall Customers Data Analysis :shopping_bags:")

# ===== Data Loading and Display =====
df = pd.read_csv("Mall_Customers.csv")
st.header("Mall Customers Dataset")
st.table(df.head())
df= df.drop(columns=['CustomerID'])
st.divider()

# Statistical Summary
st.header("Statistical Summary of Dataset")
st.table(df.describe())


# Display Gender Distribution
st.header("Gender Distribution")
st.dataframe(df.groupby(['Gender'])['Gender'].count().reset_index(name="Count"))


# Visualization of Initial Data
st.header("Initial Data Visualization")


# Summarize data for Age vs Annual Income (k$) by Gender and Total
age_income_summary_gender = df.groupby(['Age', 'Gender']).agg({'Annual Income (k$)': 'sum'}).reset_index()
age_income_summary_total = df.groupby('Age').agg({'Annual Income (k$)': 'sum'}).reset_index()

# Create multiline plot for Age vs Annual Income (k$)
fig1a = go.Figure()

color_map = {'Male': 'blue', 'Female': 'red'}

# Add lines for each gender
for gender in df['Gender'].unique():
    gender_data = age_income_summary_gender[age_income_summary_gender['Gender'] == gender]
    fig1a.add_trace(go.Scatter(x=gender_data['Age'], 
                               y=gender_data['Annual Income (k$)'],
                               mode='lines+markers',
                               name=f'Annual Income by {gender}',
                               line=dict(color=color_map[gender])))

# Add total line
fig1a.add_trace(go.Scatter(x=age_income_summary_total['Age'], 
                           y=age_income_summary_total['Annual Income (k$)'],
                           mode='lines+markers',
                           name='Total Annual Income',
                           line=dict(color='gray', dash='dash')))

fig1a.update_layout(title="Age vs Annual Income (k$) by Gender and Total",
                    xaxis_title="Age",
                    yaxis_title="Annual Income (k$)")
st.plotly_chart(fig1a, use_container_width=True)

# Summarize data for Age vs Spending Score (1-100) by Gender and Total
age_spending_summary_gender = df.groupby(['Age', 'Gender']).agg({'Spending Score (1-100)': 'sum'}).reset_index()
age_spending_summary_total = df.groupby('Age').agg({'Spending Score (1-100)': 'sum'}).reset_index()

# Create multiline plot for Age vs Spending Score (1-100)
fig1b = go.Figure()

# Add lines for each gender
for gender in df['Gender'].unique():
    gender_data = age_spending_summary_gender[age_spending_summary_gender['Gender'] == gender]
    fig1b.add_trace(go.Scatter(x=gender_data['Age'], 
                               y=gender_data['Spending Score (1-100)'],
                               mode='lines+markers',
                               name=f'Spending Score by {gender}',
                               line=dict(color=color_map[gender])))

# Add total line
fig1b.add_trace(go.Scatter(x=age_spending_summary_total['Age'], 
                           y=age_spending_summary_total['Spending Score (1-100)'],
                           mode='lines+markers',
                           name='Total Spending Score',
                           line=dict(color='gray', dash='dash')))

fig1b.update_layout(title="Age vs Spending Score (1-100) by Gender and Total",
                    xaxis_title="Age",
                    yaxis_title="Total Spending Score (1-100)")
st.plotly_chart(fig1b, use_container_width=True)

# Summarize data for Annual Income (k$) vs Spending Score (1-100) by Gender and Total
income_spending_summary_gender = df.groupby(['Annual Income (k$)', 'Gender']).agg({'Spending Score (1-100)': 'sum'}).reset_index()
income_spending_summary_total = df.groupby('Annual Income (k$)').agg({'Spending Score (1-100)': 'sum'}).reset_index()

# Create multiline plot for Annual Income (k$) vs Spending Score (1-100)
fig1c = go.Figure()

# Add lines for each gender
for gender in df['Gender'].unique():
    gender_data = income_spending_summary_gender[income_spending_summary_gender['Gender'] == gender]
    fig1c.add_trace(go.Scatter(x=gender_data['Annual Income (k$)'], 
                               y=gender_data['Spending Score (1-100)'],
                               mode='lines+markers',
                               name=f'Spending Score by {gender}',
                               line=dict(color=color_map[gender])))

# Add total line
fig1c.add_trace(go.Scatter(x=income_spending_summary_total['Annual Income (k$)'], 
                           y=income_spending_summary_total['Spending Score (1-100)'],
                           mode='lines+markers',
                           name='Total Spending Score',
                           line=dict(color='gray', dash='dash')))

fig1c.update_layout(title="Annual Income (k$) vs Spending Score (1-100) by Gender and Total",
                    xaxis_title="Annual Income (k$)",
                    yaxis_title="Total Spending Score (1-100)")
st.plotly_chart(fig1c, use_container_width=True)







# Preprocessing 
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
features = df[['Gender', 'Age', 'Annual Income (k$)']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
# Save the model
pickle.dump(kmeans, open('mall_kmeans.pkl', 'wb'))

# ===========================================

features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define range of k values
k_values = list(range(4, 9))

# Function to calculate WCSS
def calculate_wcss(features, k_values):
	wcss = []
	for k in k_values:
		kmeans = KMeans(n_clusters=k, random_state=42)
		kmeans.fit(features)
		wcss.append(kmeans.inertia_)
	return wcss

# Function to calculate Silhouette Scores
def calculate_silhouette(features, k_values):
	silhouette_scores = []
	for k in k_values:
		kmeans = KMeans(n_clusters=k, random_state=42)
		clusters = kmeans.fit_predict(features)
		score = silhouette_score(features, clusters)
		silhouette_scores.append(score)
	return silhouette_scores

# Calculate WCSS for each case
wcss = calculate_wcss(features_scaled, k_values)


# Calculate Silhouette Scores for each case
silhouette = calculate_silhouette(features_scaled, k_values)
# ===========================================
# Plot WCSS for all cases
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(k_values, wcss, marker='o', label='(Age, Income, Spending)')
ax.set_title('Elbow Method for Optimal Number of Clusters')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('WCSS')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Plot Silhouette Scores for all cases
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(k_values, silhouette, marker='o', label='(Age, Income, Spending)')
ax.set_title('Silhouette Scores for Different Number of Clusters')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Silhouette Score')
ax.legend()
ax.grid(True)
st.pyplot(fig)


# ===========================================

# KMeans clustering with k=6
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
# Save the model


df['Cluster'] = clusters
# ===== Pair Plot of Clusters =====
st.subheader("Pair Plot of Clusters")
buffer = io.BytesIO()
sns.pairplot(df, hue='Cluster', palette='tab10')
plt.title('Pair Plot for Clusters')
plt.savefig(buffer, format='png')
buffer.seek(0)
st.image(buffer)


# Calculate silhouette score for clustering
silhouette_avg = silhouette_score(features_scaled, clusters)

# PCA for dimensionality reduction
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features_scaled)

clusters_pca = kmeans.predict(features_scaled)
silhouette_pca = silhouette_score(features_scaled, clusters_pca)




# ===========================================

def plot_scatter_plotly(data, labels, title, score):
    fig = go.Figure()
    unique_labels = np.unique(labels)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']  # adjust colors if needed

    for label in unique_labels:
        mask = (labels == label)
        fig.add_trace(go.Scatter(
            x=data[mask, 0],
            y=data[mask, 1],
            mode='markers',
            marker=dict(size=10, opacity=0.8),
            name=f'Cluster {label}'
        ))

    fig.update_layout(
        title=f'{title} (Silhouette Score: {score:.2f})',
        xaxis_title='Income',
        yaxis_title='Spending Score',
        legend_title='Clusters',
        template='plotly_white'
    )
    return fig
	
# Scatter plot before clustering
fig_before = go.Figure()
fig_before.add_trace(go.Scatter(
	x=features_scaled[:, 0],
	y=features_scaled[:, 1],
	mode='markers',
	marker=dict(size=10, color='grey', opacity=0.5),
	name='Data Points'
))
fig_before.update_layout(
	title='Before Clustering',
	xaxis_title='Income',
	yaxis_title='Spending Score',
	template='plotly_white'
)
st.plotly_chart(fig_before)

# Scatter plot after clustering
fig_after = plot_scatter_plotly(features_scaled, clusters, 'After Clustering', silhouette_avg)
st.plotly_chart(fig_after)

# Scatter plot after PCA clustering
fig_pca = plot_scatter_plotly(features_pca, clusters_pca, 'After PCA Decomposition', silhouette_pca)
st.plotly_chart(fig_pca)

# ===========================================


def plot_3d_scatter(features_pca, clusters_pca):
    fig = go.Figure()

    # Add 3D scatter plot
    fig.add_trace(go.Scatter3d(
        x=features_pca[:, 1],
        y=features_pca[:, 2],
        z=features_pca[:, 0],
        mode='markers',
        marker=dict(
            size=5,
            color=clusters_pca,  # Color by cluster
            colorscale='viridis',  # Color map
            opacity=0.8
        ),
        text=[f'Cluster {c}' for c in clusters_pca]  # Add text labels for hover
    ))

    # Update layout
    fig.update_layout(
        title='K-means Clustering: 3D Plot',
        scene=dict(
            xaxis_title='Annual Income',
            yaxis_title='Score',
            zaxis_title='Age'
        ),
        template='plotly_white'
    )

    return fig

# Plotting
st.write("3D Scatter Plot")

# Plot 3D scatter plot using Plotly
fig_3d = plot_3d_scatter(features_pca, clusters_pca)
st.plotly_chart(fig_3d)


# ===========================================


# ===== Evaluation Metrics =====
st.subheader("Evaluation Metrics")
silhouette_avg = silhouette_score(features_scaled, clusters)
davies_bouldin = davies_bouldin_score(features_scaled, clusters)
calinski_harabasz = calinski_harabasz_score(features_scaled, clusters)

st.write(f"***Silhouette Score:*** {silhouette_avg:.2f}")
st.write(f"***Davies-Bouldin Score:*** {davies_bouldin:.2f}")
st.write(f"***Calinski-Harabasz Score:*** {calinski_harabasz:.2f}")

st.divider()
