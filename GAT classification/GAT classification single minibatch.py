import numpy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, GCNConv, GATConv ,Set2Set
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import DataListLoader
import random
import os

from  funtion import  findangel,findratio,findnorm,findmaxdot
from torch_geometric.data import DataLoader as G_DataLoader

# x = torch.tensor([ [0, 1, 0],
#                    [1, 0, 0],
#                    [0, 0, 1] ], dtype=torch.float)
# edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
# edge_attr = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float)
# y = torch.tensor([[0]], dtype=torch.float)  # 目标标签 (0: 无毒, 1: 有毒)
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
from torch.utils.data import Dataset

num_intervals = 10
interval_width = 1 / num_intervals


class DiabetesDataset(Dataset):
    def __init__(self, filepath , num=31 , row = 31):
        super(DiabetesDataset, self).__init__()
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0] // row
        self.num_nodes = num
        self.data = []

        for i in range(self.len):
            # x_data = torch.from_numpy(xy[i * row:i * row + num, [7,8,9,10,13,14,15,16,17,18]])
            x_data = torch.from_numpy(xy[i * row:i * row + num, [3,7,4,8,5,9,6,10,11,13,12,14,15,16,17,18]])

            # x_data = torch.from_numpy(xy[i * row:i * row + num, [3,7,4,8,5,9,6,10,11,13,12,14,15,16,17]])



            # print("angel", x_data[:,14])
            # findmaxdot(x_data[:,12])  # 运动一致性
            # findnorm(x_data[:,13])    # 运动模长比
            # findmaxdot(x_data[:,14],10,0)  # 运动角度
            # findratio(x_data[:,15],10)   #  邻居距离

            # x_data = torch.from_numpy(xy[i * 51:i * 51 + num, [3,7,4,8,5,9,6,10,11,13,12,14,15]])
            weight = torch.from_numpy(xy[i * row:i * row + num, 19:19+num]) - torch.eye(num, dtype=torch.float32)
            weight[:,1:] = 0
            # weight[1:,1:] = 0

            edge_index = torch.where(weight > 0.4)
            edge_index = torch.stack(edge_index)
            edge_attr = weight[edge_index[0], edge_index[1]]
            # edge_attr = edge_attr.view(-1, 1)

            y_data = torch.tensor([xy[i * row, 0]], dtype=torch.float)

            self.data.append(Data(x=x_data, edge_index=edge_index, edge_attr=edge_attr, y=y_data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



#


path = './带有尺度/last/'
# path = './带有尺度 3000/last/'
# path = './小样本 - 副本/'
# path = './fisheyeall/'
files = os.listdir(path)
files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
data = []
datalens = []
for file in files:
    file_path = os.path.join(path, file)
    dataset = DiabetesDataset(file_path)
    print(file, len(dataset))
    datalens.append(len(dataset))
    data.extend(dataset.data)

batch_size = 20000

count_zero = 0
count_one = 0

for datait in data:
    if datait.y.item() == 0:
        count_zero += 1
    elif datait.y.item() == 1:
        count_one += 1

print("Number of samples with target 0:", count_zero)
print("Number of samples with target 1:", count_one)


print(sum(datalens[:100]))
train_dataset = data[sum(datalens[:200]):]

# test_dataset1 = data[sum(datalens[:400]):sum(datalens[:600])]
# test_dataset2 = data[sum(datalens[:200]):sum(datalens[:400])]

test_dataset3 = data[:sum(datalens[:200])]


positive_samples = [sample for sample in train_dataset if sample.y.item() == 1]
negative_samples = [sample for sample in train_dataset if sample.y.item() == 0]

min_samples = min(len(positive_samples), len(negative_samples))

positive_samples = random.sample(positive_samples, min_samples)
negative_samples = random.sample(negative_samples, min_samples)

balanced_data = positive_samples + negative_samples




print(len(positive_samples), len(negative_samples))
print(" lens of train ", len(balanced_data))
# print(" lens of test ", len(test_dataset1), len(test_dataset2), len(test_dataset3))

# train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataListLoader(balanced_data, batch_size=batch_size, shuffle=True)
# test_loader1 = DataListLoader(test_dataset1, batch_size=batch_size, shuffle=False)
# test_loader2 = DataListLoader(test_dataset2, batch_size=batch_size, shuffle=False)
test_loader3 = DataListLoader(test_dataset3, batch_size=batch_size, shuffle=False)

# train_dataset = [data[i] for i in range(int(len(data) * 0.8))]
# test_dataset = [data[i] for i in range(int(len(data) * 0.8), len(data))]


from torch.utils.data import random_split


#

# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# print(train_loader)
# for train in len(train_loader):
#     print(train)
#     print(train)
from torch_geometric.nn import global_add_pool, SAGEConv , global_max_pool , global_mean_pool
import torch.nn as nn
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(GraphSAGE, self).__init__()
        self.sage_layers = nn.ModuleList()
        self.sage_layers.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, 1)
        self.bn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        for layer, bn_layer in zip(self.sage_layers, self.bn_layers):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = bn_layer(x)

        x = global_add_pool(x, batch=batch)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

from torch_scatter import scatter_add
from torch.nn import Linear
from torch_geometric.utils import softmax

import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_scatter import scatter_add



#
class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(GIN, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.gin_layers.append(GINConv(nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )))

        for _ in range(num_layers - 1):
            self.gin_layers.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )))
        self.set2set = Set2Set(hidden_dim, processing_steps=3)


        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim / 4))
        self.fc2 = nn.Linear(int(hidden_dim / 4), 2)

        self.bn = nn.BatchNorm1d(int(hidden_dim / 4))
        self.bn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        self.attention = torch.nn.Linear(hidden_dim, 1)

    def graph_attention_pool(self, x, batch):


        attention_scores = self.attention(x)
        # print(attention_scores)
        attention_scores = softmax(attention_scores, batch)

        pooled_features = scatter_add(x * attention_scores, batch, dim=0)

        return pooled_features


    def forward(self, x, edge_index, edge_attr, batch):
        for layer, bn_layer in zip(self.gin_layers, self.bn_layers):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = bn_layer(x)

        x = self.graph_attention_pool(x, batch)  # 注意力池化
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn(x)

        x = self.fc2(x)

        return x



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, head=4, fc = 40):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=head )  #
        self.conv2 = GATConv(hidden_channels * head, hidden_channels, heads=head)  #
        self.conv3 = GATConv(hidden_channels * head, hidden_channels, heads=head)  #

        self.set2set = Set2Set(hidden_channels * head, processing_steps=3)

        self.fc1 = torch.nn.Linear(hidden_channels* head, fc)
        # self.fc1 = torch.nn.Linear(hidden_channels* head *2, fc)   #for set2set
        self.fc2 = torch.nn.Linear(fc, fc)
        self.fc3 = torch.nn.Linear(fc, 1)

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * head)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * head)
        self.bn25 = torch.nn.BatchNorm1d(hidden_channels * head)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels* head )

        self.bn4 = torch.nn.BatchNorm1d(fc)
        self.bn5 = torch.nn.BatchNorm1d(fc)

        self.dropout = torch.nn.Dropout(0.6)
        self.attention = torch.nn.Linear(hidden_channels * head, 1)

    def graph_attention_pool(self, x, batch):
        attention_scores = self.attention(x)
        attention_scores = softmax(attention_scores, batch)
        pooled_features = scatter_add(x * attention_scores, batch, dim=0)
        return pooled_features

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = self.bn25(x)
        x = x.relu()
        x = self.graph_attention_pool(x, batch)  # 注意力池化
        # x = self.set2set(x, batch)

        # x = global_add_pool(x, batch=batch)
        # x = self.bn3(x)

        # x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = x.relu()

        x = self.fc2(x)
        x = self.bn5(x)
        x = x.relu()

        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x



from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool




hidden_channels = 32
fc = 120

from torch.optim.lr_scheduler import StepLR

# 初始化模型并定义优化器
# model = GIN( in_dim = 16, hidden_dim = hidden_channels, num_layers = 3)
# model = GraphSAGE( in_dim = 16, hidden_dim = hidden_channels, num_layers = 3)
model = GAT( in_channels = 16, hidden_channels= hidden_channels, head=4, fc = 48)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# #
my_net = model.to(device)





criterion = nn.BCELoss()

# sigmoid
def train(model, optimizer, data_loader):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        batch = Batch.from_data_list(data)
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        batch.y = batch.y.squeeze()
        # loss = F.binary_cross_entropy(out.view(-1), batch.y.view(-1))
        loss = criterion(out.view(-1), batch.y.view(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)

    return average_loss

def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    TPFP = 0
    TP = 0
    TPFN = 0
    with torch.no_grad():
        for data in data_loader:
            batch = Batch.from_data_list(data)
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = torch.round(out)
            # correct += int((torch.abs(pred.view(-1)
            #                           - batch.y.view(-1)) < 1e-1).sum())
            correct += int(torch.eq(pred.view(-1), batch.y.view(-1)).sum())
            TPFP += int((torch.abs(1 - pred.view(-1)) < 1e-1).sum())
            # 注意张量尺寸

            TP += int((torch.logical_and(torch.eq(pred.view(-1), 1)
                                         , torch.eq(batch.y.view(-1), 1))).sum())


            # pred_zero_batch_one = torch.logical_and(torch.eq(pred.view(-1), 0), torch.eq(batch.y.view(-1), 1))
            # values_pred_zero_batch_one = out[pred_zero_batch_one]
            #

            # pred_one_batch_zero = torch.logical_and(torch.eq(pred.view(-1), 1), torch.eq(batch.y.view(-1), 0))
            # values_pred_one_batch_zero = out[pred_one_batch_zero]


            # print("Values where pred is 0 and batch is 1:", values_pred_zero_batch_one)
            # print("Values where pred is 1 and batch is 0:", values_pred_one_batch_zero)

            # TP = int(torch.eq(torch.eq(pred, 1), torch.eq(batch.y, 1)).sum())
            # TP += int(    (torch.logical_and(  (torch.abs(1- batch.y.view(-1)) < 1e-1)
            #               ,(torch.abs(1 - pred.view(-1)) < 1e-1) )
            #                ).sum())
            TPFN += int( (torch.eq(batch.y.view(-1) ,1 )).sum() )
            # TPFN += int((1 - batch.y.view(-1) < 1e-1).sum())
            # print("TP", TP, "TPFN", TPFN, "TPFP", TPFP)
            total += len(batch.y)
    if TPFP != 0:
        precision = TP / TPFP
    else:
        precision = 0.0
    acc = correct / total

    recall = TP / TPFN
    return [precision,recall]
#




# # folat16 32
# def train(model, optimizer, data_loader, scaler):
#     model.train()
#     total_loss = 0
#     for data in data_loader:
#         optimizer.zero_grad()
#         batch = Batch.from_data_list(data)
#         batch = batch.to(device)
#
#
#         with autocast():
#             out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
#             loss = torch.nn.CrossEntropyLoss()(out, batch.y.long())
#
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#
#         total_loss += loss.item()
#
#     average_loss = total_loss / len(data_loader)
#     return average_loss
#
# #

#
#  #for CrossEntropyLoss
# def train(model, optimizer, data_loader):
#     model.train()
#     total_loss = 0
#     for data in data_loader:
#         optimizer.zero_grad()
#         batch = Batch.from_data_list(data)
#         batch = batch.to(device)
#         # transformed_labels = preprocess_labels(batch.y)
#         out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
#         # batch.y = batch.y.unsqueeze(1)
#         # print(out[:20])
#         # print(batch.y[:20])
#         loss = torch.nn.CrossEntropyLoss()(out, batch.y.long())
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
# #
#     average_loss = total_loss / len(data_loader)
#
#     return average_loss
#
#
# def test(model, data_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     TP = 0
#     TPFP = 0
#     TPFN = 0
#     with torch.no_grad():
#         for data in data_loader:
#             batch = Batch.from_data_list(data)
#             batch = batch.to(device)
#             out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
#             pred = torch.argmax(out, dim=1)
#             batch.y = batch.y.squeeze().long()  #
#             correct += int(torch.sum(pred == batch.y).item())
#             total += len(batch.y)
#
#             TPFP += int((torch.eq(pred , 1)).sum())
#             # 注意张量尺寸
#             TPFN += int((1 == batch.y.view(-1)).sum())
#
#             TP += int((torch.logical_and(torch.eq(pred, 1)
#                                          , torch.eq(batch.y.view(-1), 1))).sum())
#             # print("TP", TP, "TP FN", TPFN, "TP FP", TPFP)
#         if TPFP != 0:
#             precision = TP / TPFP
#         else:
#             precision = 0.0
#         recall = TP / TPFN
#     acc = correct / total
#     return [precision, recall]

last = 0
scaler = GradScaler()
losses = []
train_precision = []
train_recall = []
# for epoch in range(1, 8000):
#     loss  = train(model, optimizer, train_loader, scaler)
for epoch in range(1, 500):
    loss  = train(model, optimizer, train_loader)
    train_p = test(model, train_loader)
    # test_p1 = test(model, test_loader1)
    # test_p2 = test(model, test_loader2)

    test_p3 = test(model, test_loader3)
    print("epoch", epoch , "loss", loss)
    print("Train precision", train_p)
    # print("test1  precision", test_p1)
    # print("test2  precision", test_p2)
    print("test3  precision", test_p3)
    scheduler.step()
    losses.append(loss)
    train_precision.append(train_p[0])
    train_recall.append(train_p[1])

import pandas as pd
df = pd.DataFrame({
    'Epoch': list(range(1, 500)),
    'Loss': losses,
    'Precision': train_precision,
    'Recall': train_recall
})

df.to_excel('32 48.xlsx', index=False)