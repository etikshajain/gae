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

def col865_data():
    df1=pd.read_csv('../data/product_detail_fin.csv')
    df2=pd.read_csv('../data/ppv_encrypted-001.csv', nrows=5000)
    df3=df1.merge(df2,how='inner',on='product_id')
    ohe = pd.get_dummies(data=df3, columns=['cms_vertical'])
    mean_rating = df2.groupby("account_id_enc")["count"].mean().rename("mean")
    num_rating = df2.groupby("account_id_enc")["product_id"].count().rename("total")

    ## Maps for edges and id encoding
    user_dict={}
    prod_dict={}
    edge_list=[]
    s=[]
    t=[]
    prod_features=[]
    user_features=[]
    e_wt=[]

    prod_it=0
    user_it=df3['product_id'].nunique()

    for index,row in ohe.iterrows():
        prod_id=row['product_id']
        user_id=row['account_id_enc']

        if (prod_id in prod_dict.keys() and user_id in user_dict.keys()):
            print()
        elif (prod_id in prod_dict.keys()):
            user_features.append(np.array([mean_rating[user_id], num_rating[user_id]]))
            user_dict[user_id]=user_it
            user_it+=1
        elif (user_id in user_dict.keys()):
            prod_features.append(ohe.iloc[index].drop(['product_id','account_id_enc','count']).to_numpy())
            prod_dict[prod_id]=prod_it
            prod_it+=1

        else:
            user_features.append(np.array([mean_rating[user_id], num_rating[user_id]]))
            prod_features.append(ohe.iloc[index].drop(['product_id','account_id_enc','count']).to_numpy())
            prod_dict[prod_id]=prod_it
            prod_it+=1
            user_dict[user_id]=user_it
            user_it+=1
        s.append(user_dict[user_id])
        t.append(prod_dict[prod_id])
        s.append(prod_dict[prod_id])
        t.append(user_dict[user_id])
        e_wt.append(row['count'])
        e_wt.append(1) #**************************************************
        edge_list.append([user_dict[user_id],prod_dict[prod_id]])
        # edge_list.append([prod_dict[prod_id],user_dict[user_id]])

    source=np.asarray(s)
    target=np.asarray(t)
    wts=np.asarray(e_wt)
    sp_mat = sp.coo_matrix((wts, (source, target)))
    g = dgl.from_scipy(sp_mat)
    G_nx = g.to_networkx()
    adj = nx.adjacency_matrix(G_nx)

    # Fit the PCA model to your data and transform your features
    # reduced_features = pca.fit_transform(np.asarray(prod_features))
    # reduced_features = torch.tensor(reduced_features, dtype=torch.float32, requires_grad=True)
    prod_features = torch.tensor(prod_features, dtype=torch.float32, requires_grad=True)
    user_features = torch.tensor(user_features, dtype=torch.float32, requires_grad=True)
    user_features = torch.randn(user_features.shape[0], prod_features.shape[1])
    user_features = torch.tensor(user_features, dtype=torch.float32, requires_grad=True)
    # expander = FeatureExpander(2, prod_features.shape[1])
    # user_features_expanded = expander(user_features)
    # user_features_expanded = torch.tensor(user_features_expanded, dtype=torch.float32, requires_grad=True)
    node_features = torch.cat([user_features, prod_features], dim=0) # **********************************************
    features=sp.csr_matrix(node_features.detach().numpy()).tolil()
    return adj, features


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
