import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt

# Route Optimization (Dijkstra's Algorithm)
def find_shortest_path(graph, source, destination):
    path = nx.dijkstra_path(graph, source, destination)
    return path

# Location Clustering (k-Means Clustering)
def cluster_locations(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    data['Cluster'] = kmeans.fit_predict(data[['Latitude', 'Longitude']])
    return data, kmeans

# Real-time Search using Trie
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_end_of_word = True
    
    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._collect_all_words(node, prefix)
    
    def _collect_all_words(self, node, prefix):
        words = []
        if node.is_end_of_word:
            words.append(prefix)
        for char, child in node.children.items():
            words.extend(self._collect_all_words(child, prefix + char))
        return words

# Data Visualization
def visualize_clusters(data):
    fig, ax = plt.subplots()
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f"Cluster {cluster}")
    ax.legend()
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("Smart Route Planner and Location Recommender")
    
    # Route Optimization
    st.header("Route Optimization")
    st.text("Enter source and destination nodes:")
    source = st.text_input("Source")
    destination = st.text_input("Destination")
    if st.button("Find Route"):
        graph = nx.Graph()
        graph.add_weighted_edges_from([
            ('A', 'B', 1), ('B', 'C', 2), ('A', 'C', 2), ('C', 'D', 1)
        ])
        path = find_shortest_path(graph, source, destination)
        st.write(f"Shortest path: {path}")
    
    # Location Clustering
    st.header("Location Clustering")
    st.text("Upload a dataset with Latitude and Longitude columns.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        num_clusters = st.slider("Number of Clusters", 2, 10, 3)
        clustered_data, kmeans = cluster_locations(data, num_clusters)
        st.write(clustered_data)
        visualize_clusters(clustered_data)
    
    # Real-time Search
    st.header("Real-time Search")
    st.text("Enter a prefix to search for matching locations.")
    trie = Trie()
    locations = ["Paris", "London", "Los Angeles", "Lagos", "Lisbon"]
    for loc in locations:
        trie.insert(loc)
    query = st.text_input("Search Prefix")
    if query:
        results = trie.search_prefix(query)
        st.write(f"Results: {results}")
    
    # Data Visualization
    st.header("Data Visualization")
    st.text("Interactive visualizations of clustered locations or routes.")
    # Add any custom visualizations here.

if __name__ == "__main__":
    main()
