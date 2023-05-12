import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
import networkx as nx
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

data = pd.read_csv(
    "/home/zattia/ukbb_analyzer-1/Data/snp_snp_network.csv",index_col=0)  # SNPAssoc data
from_node = []
to_node = []
intensity = []
typeEdge = []
for i in range(len(data.columns)):
    for j in range(i+1, len(data.columns)):
        p = data.iloc[j, i]
        if p<0.05:
            from_node.append(data.columns[i])
            to_node.append(data.columns[j])
            intensity.append(p)
            typeEdge.append('Undirected')
print(min(intensity))
intensity = min(intensity)/np.array(intensity)
edges_df = pd.DataFrame()
edges_df['Source'] = from_node
edges_df['Target'] = to_node
edges_df['Type'] = typeEdge
edges_df['Weight'] = intensity
edges_df.to_csv('/home/zattia/ukbb_analyzer-1/Data/edges.csv', index=False)

unique = dict()
 
for x in from_node:
    unique[x] = unique.get(x, 0) + 1


for x in to_node:
    unique[x] = unique.get(x, 0) + 1

unique_list = list(unique.keys())
size = list(unique.values())
size = (np.array(size)/max(size))*300
intensity = min(intensity)/np.array(intensity)

nodes_df = pd.DataFrame()
nodes_df['Id'] = unique_list
nodes_df['Label'] = unique_list
nodes_df.to_csv('/home/zattia/ukbb_analyzer-1/Data/nodes.csv', index=False)

G = nx.Graph()
for i in range(len(unique_list)):
    G.add_node(unique_list[i])

for i in range(len(from_node)):
    G.add_edge(from_node[i], to_node[i])

deg_centrality = nx.degree_centrality(G)
sorted_deg_centrality = sorted(deg_centrality.items(), key=lambda x:x[1], reverse=True)
converted_dict = dict(sorted_deg_centrality)
print(converted_dict)

from networkx.algorithms.community.label_propagation import label_propagation_communities

communities = label_propagation_communities(G)
print([community for community in communities])

nx.draw(G, node_size=size, width=intensity) # with_labels = True
plt.savefig("/home/zattia/ukbb_analyzer-1/networkFig.png")