import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import dgl
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from scipy.linalg import svd
from surprise import SVD
from surprise import Dataset
from surprise import Reader

class FeatureExpander(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureExpander, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Initialize the PCA model
pca = PCA(n_components=2)  # Reduce to 2 dimensions


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def create_node_dict(df_merged):
    product_ids = np.unique(df_merged['product_id'])
    user_ids = np.unique(df_merged['account_id_enc'])
    node_dict = {}
    for i in range(len(product_ids)):
        node_dict[product_ids[i]] = i 
    for i in range(len(user_ids)):
        node_dict[user_ids[i]] = i + len(product_ids)
    return node_dict

def col865_data():
    df1=pd.read_csv('../data/product_detail_fin.csv')
    df2=pd.read_csv('../data/ppv_encrypted-001.csv', nrows=100000)
    df3=df1.merge(df2,how='inner',on='product_id')

    # frequency encoding
    # Calculate the frequency of each category
    frequency_map = df3['cms_vertical'].value_counts().to_dict()

    # Map the frequency values to the original column
    df3['cms_vertical'] = df3['cms_vertical'].map(frequency_map)

    # delete sparse data
    # dropping rows with high cms and price
    df3 = df3[df3['cms_vertical'] < 600]
    df3 = df3[df3['price'] < 3000]
    df3=df3.sample(frac=1).reset_index(drop=True)

    # Normalising
    min_val = df3['count'].min()
    max_val = df3['count'].max()
    df3['count'] = (df3['count'] - min_val) / (max_val - min_val)
    min_val = df3['price'].min()
    max_val = df3['price'].max()
    df3['price'] = (df3['price'] - min_val) / (max_val - min_val)

    # ohe = pd.get_dummies(data=df3, columns=['cms_vertical'])
    ohe = df3
    # User features
    mean_rating = df2.groupby("account_id_enc")["count"].mean().rename("mean")
    num_rating = df2.groupby("account_id_enc")["product_id"].count().rename("total")

    ## Maps for edges and id encoding
    node_dict=create_node_dict(df3)
    s=[]
    t=[]
    e_wt=[]
    edge_list = []

    # create empty features 2d array of size (total_nodes, feature dim)
    for index,row in ohe.iterrows():
        prod_feat_len = len(ohe.iloc[index].drop(['product_id','account_id_enc','count']).to_numpy())
        break
    total_nodes = len(node_dict)
    prod_feat_len = 64
    node_features = np.empty(shape=(total_nodes,prod_feat_len))

    for index,row in ohe.iterrows():
        prod_id=row['product_id']
        user_id=row['account_id_enc']
        e_wt.append(row['count'])
        e_wt.append(0)

        user_index = node_dict[user_id]
        prod_index = node_dict[prod_id]
        s.append(node_dict[user_id])
        t.append(node_dict[prod_id])
        s.append(node_dict[prod_id])
        t.append(node_dict[user_id])
        edge_list.append((user_index, prod_index))
        edge_list.append((prod_index, user_index))

        # node_features[prod_index] = ohe.iloc[index].drop(['product_id','account_id_enc','count']).to_numpy()
        u=np.array([mean_rating[user_id], num_rating[user_id]])
        padded_u = np.pad(u, (0, 62), mode='constant')
        p = ohe.iloc[index].drop(['product_id','account_id_enc','count']).to_numpy()
        padded_p = np.pad(p, (0, 62), mode='constant')
        # padded_u = u
        node_features[user_index] = padded_u
        node_features[prod_index] = padded_p

    source=np.asarray(s)
    target=np.asarray(t)
    wts=np.asarray(e_wt)
    sp_mat = sp.coo_matrix((wts, (source, target)))
    g = dgl.from_scipy(sp_mat)
    G_nx = g.to_networkx()
    adj = nx.adjacency_matrix(G_nx)

    adjacency_matrix = np.zeros((total_nodes, total_nodes))
    # Fill the adjacency matrix with weights
    i=0
    for edge in edge_list:
        node1, node2 = edge

        weight=e_wt[i]
        # print(weight)
        i+=1
        adjacency_matrix[node1][node2] = weight

    # Print the weighted adjacency matrix
    print(adjacency_matrix)
    print(node_features.shape)

    # print(df1.count())
    # print(df2.count())
    # print(df3.count())
    # print(len(s))
    # print(len(t))
    # print(len(e_wt))
    # print(len(node_features))
    # print(len(node_features[0]))

    features=sp.csr_matrix(node_features).tolil()
    return adj, features, node_features


def load_data(dataset):
    if dataset == 'col865':
        return col865_data()
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("gae/data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("gae/data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    print(type(allx))
    print(type(tx))
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print('******************************************************************************************')
    print(type(features))
    # print(type(adj))
    return adj, features
