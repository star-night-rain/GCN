import numpy as np
import pandas as pd
import time

import torch

def accuracy(output,labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct/len(labels)

#除以节点度数
def normalize1(mx):
    rowsum = np.array(mx.sum(axis=1),dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    mx = np.dot(r_mat_inv, mx)
    return mx

#拉普拉斯平滑
def normalize2(mx):
    rowsum = np.array(mx.sum(axis=1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    mx = np.dot(r_mat_inv, mx)
    mx = np.dot(mx,r_mat_inv)
    return mx



def load_data():
    print("开始加载数据......")
    start = time.perf_counter()

    cora_content = pd.read_csv("dataset/cora/cora.content", sep="\t", header=None)
    cora_cities = pd.read_csv("dataset/cora/cora.cites", sep="\t", header=None)
    labels = cora_content.iloc[:, -1]
    labels = pd.get_dummies(labels)
    labels = np.array(labels)
    labels = torch.LongTensor(np.where(labels)[1])


    content_idx = list(cora_content.index)
    paper_idx = list(cora_content.iloc[:, 0])
    mp = dict(zip(paper_idx, content_idx))


    features = cora_content.iloc[:, 1:-1]
    features = normalize1(features)
    features = np.array(features)
    features = torch.FloatTensor(features)

    adj_size = cora_content.shape[0]
    adj = np.zeros((adj_size, adj_size))
    for i, j in zip(cora_cities[0], cora_cities[1]):
        x = mp[i]
        y = mp[j]
        adj[x][y] = 1
        adj[y][x] = 1
    adj = np.array(adj)
    eye = np.identity(adj.shape[0])
    adj = adj+eye

    adj = normalize1(adj)
    adj = torch.FloatTensor(adj)

    train_index = range(140)
    valid_index = range(200,500)
    test_index = range(500,1500)

    train_index = torch.LongTensor(train_index)
    valid_index = torch.LongTensor(valid_index)
    test_index = torch.LongTensor(test_index)

    end = time.perf_counter()
    print("加载数据一共花费了：{:.4f}s".format(end - start))

    return features,labels,adj,train_index,valid_index,test_index

