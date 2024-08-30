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

# ===== Data Preprocessing =====
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# ===== Finding Optimal Number of Clusters =====

x_df = df[['Gender', 'Age', 'Annual Income (k$)']]
# x_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
#x_df = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
scaled_df = scaler.fit_transform(x_df)

st.subheader("Finding Optimal Number of Clusters")
# Elbow Method to Determine Optimal K
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=30, random_state=0)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

kdf = pd.DataFrame({'k': k_range, 'wcss': wcss})
fig2 = px.line(kdf, x="k", y="wcss", title="Elbow Method for Optimal K")
st.plotly_chart(fig2, use_container_width=True)

# ===== Applying K-Means Clustering =====
st.subheader("K-Means Clustering")


optimal_k = 5


kmeans_model = KMeans(n_clusters=optimal_k, init="k-means++", max_iter=30, random_state=42, tol=0.001)
clusters = kmeans_model.fit_predict(scaled_df)
df['Cluster'] = clusters

# Save the model
pickle.dump(kmeans_model, open('mall_kmeans.pkl', 'wb'))















# ===== Pair Plot of Clusters =====
st.subheader("Pair Plot of Clusters")
buffer = io.BytesIO()
sns.pairplot(df, hue='Cluster', palette='tab10')
plt.title('Pair Plot for Clusters')
plt.savefig(buffer, format='png')
buffer.seek(0)
st.image(buffer)



# ==== K-Means and PCA 
# Perform KMeans clustering
x = df.drop(['Spending Score (1-100)'], axis=1)
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", max_iter=30, random_state=0)
df['y1'] = kmeans.fit_predict(x)

# Apply PCA
st.header("Applying PCA on features")
x_scaled = scale(x)
pca = PCA(n_components=4)
pca_x = pca.fit_transform(x_scaled)
df['y2'] = kmeans.fit_predict(pca_x)

# Visualize the clustering results without PCA
st.header("Visualizing the labels without PCA and clusters")
fig2 = go.Figure()

# Adding trace for scatter plot without PCA
fig2.add_trace(go.Scatter(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    mode='markers',
    marker=dict(
        size=df['Age'],  # Size based on Age
        color=df['y1'],  # Color based on clusters
        colorscale='Viridis',  # Color scale for clusters
        colorbar=dict(title='Cluster', tickvals=list(range(5)), ticktext=[f'Cluster {i}' for i in range(5)])
    ),
    text=df['Age'],  # Tooltip text
    hoverinfo='all',
    name='Without PCA'
))

fig2.update_layout(
    title="Clusters Without PCA",
    xaxis_title='Annual Income (k$)',
    yaxis_title='Spending Score (1-100)',
)

st.plotly_chart(fig2, use_container_width=True)

# Visualize the clustering results with PCA
st.header("Visualizing the labels with PCA and clusters")
fig3 = go.Figure()

# Adding trace for scatter plot with PCA
fig3.add_trace(go.Scatter(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    mode='markers',
    marker=dict(
        size=df['Age'],  # Size based on Age
        color=df['y2'],  # Color based on clusters after PCA
        colorscale='Viridis',  # Color scale for clusters
        colorbar=dict(title='Cluster', tickvals=list(range(5)), ticktext=[f'Cluster {i}' for i in range(5)])
    ),
    text=df['Age'],  # Tooltip text
    hoverinfo='all',
    name='With PCA'
))

fig3.update_layout(
    title="Clusters With PCA",
    xaxis_title='Annual Income (k$)',
    yaxis_title='Spending Score (1-100)',
)

st.plotly_chart(fig3, use_container_width=True)

# Evaluation Metrics
c1, c2 = st.columns(2)
c1.subheader("Silhouette Score without PCA")
c1.write(silhouette_score(x, df['y1']))
c2.subheader("Silhouette Score with PCA")
c2.write(silhouette_score(x_scaled, df['y2'])) 






# ============== Visualization Before and After Clustering ======
colors_list = px.colors.sequential.Plasma  # Example continuous color scale

st.header("Visualization of Clustering Results")

# Before Clustering
st.subheader("Before Clustering")
fig3 = go.Figure()

# Adding trace for scatter plot with color scale
fig3.add_trace(go.Scatter(
	
	x=df['Annual Income (k$)'],
	y=df['Spending Score (1-100)'],
	
    mode='markers',
    marker=dict(
        size=10,  # Fixed size for all markers or adjust based on your requirement
        color=df['Spending Score (1-100)'],  # Map this column to the color scale
        colorscale=colors_list,  # Apply the color scale
        colorbar=dict(title='Spending Score (1-100)', tickvals=[0, 50, 100], ticktext=['Low', 'Medium', 'High']),
        showscale=True
    ),
    text=df['Gender'],
    hoverinfo='all',  # Show hover info with color scale
    name='Before Clustering'
))

fig3.update_layout(
    #title="Before Clustering",
    xaxis_title='Annual Income (k$)',
    yaxis_title='Spending Score (1-100)',
)

st.plotly_chart(fig3, use_container_width=True)



st.subheader("After Clustering")
colors_list = px.colors.sequential.Jet 
fig5 = go.Figure()
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    fig5.add_trace(go.Scatter(
        x=cluster_data['Annual Income (k$)'],
        y=cluster_data['Spending Score (1-100)'],
        mode='markers',
        name=f'Cluster {i+1}',
        marker=dict(color=colors_list[i % len(colors_list)], size=10)
    ))

fig5.update_layout(title=f'KMeans Clustering with {optimal_k} Clusters',
                   xaxis_title='Annual Income (k$)',
                   yaxis_title='Spending Score (1-100)')
st.plotly_chart(fig5, use_container_width=True)



# ===== Evaluation Metrics =====
st.subheader("Evaluation Metrics")
silhouette_avg = silhouette_score(scaled_df, clusters)
davies_bouldin = davies_bouldin_score(scaled_df, clusters)
calinski_harabasz = calinski_harabasz_score(scaled_df, clusters)

st.write(f"***Silhouette Score:*** {silhouette_avg:.2f}")
st.write(f"***Davies-Bouldin Score:*** {davies_bouldin:.2f}")
st.write(f"***Calinski-Harabasz Score:*** {calinski_harabasz:.2f}")


