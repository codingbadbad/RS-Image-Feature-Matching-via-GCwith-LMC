import cv2
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from  funtion import  findangel,findratio,findnorm,findmaxdot
from torch_scatter import scatter_add
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, GCNConv , global_max_pool
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import DataListLoader
from torch_geometric.nn import TopKPooling
import os
import numpy as np
import torch
import torch_geometric.nn as gnn
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool, GCNConv
from torch_geometric.data import DataLoader
import random
import os
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

import numpy as np
import torch
from torch.utils.data import Dataset

class DiabetesDataset(Dataset):
    def __init__(self, filepath , num=31 , row = 31):
        super(DiabetesDataset, self).__init__()
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0] // row
        self.num_nodes = num
        self.data = []

        for i in range(self.len):
            # x_data = torch.from_numpy(xy[i * 51:i * 51 + 21, [7,8,9,10,13,14,15,16,17]])
            x_data = torch.from_numpy(xy[i * row:i * row + num, [3,7,4,8,5,9,6,10,11,13,12,14,15,16,17,18]])
            # x_data = torch.from_numpy(xy[i * row:i * row + num, [3,7,4,8,5,9,6,10,11,13,12,14,15,16,17]])



            # # print("angel", x_data[:,14])
            # findmaxdot(x_data[:,12])  # 运动一致性
            # # findnorm(x_data[:,13])    # 运动模长比
            # # findmaxdot(x_data[:,14],10,0)  # 运动角度
            # findratio(x_data[:,15],15)   #  邻居距离

            # x_data = torch.from_numpy(xy[i * 51:i * 51 + num, [3,7,4,8,5,9,6,10,11,13,12,14,15]])
            weight = torch.from_numpy(xy[i * row:i * row + num, 19:19+num]) - torch.eye(num, dtype=torch.float32)
            weight[:,1:] = 0

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class ToxicityClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, middle=240, fc=40, out=2):
        super(ToxicityClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, middle)
        self.conv2 = GCNConv(middle, middle)
        self.conv3 = GCNConv(middle, middle)

        self.fc1 = torch.nn.Linear(middle, fc)
        self.fc2 = torch.nn.Linear(fc, fc)
        self.fc3 = torch.nn.Linear(fc, out)

        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.bn0 = torch.nn.BatchNorm1d(middle)
        self.bn1 = torch.nn.BatchNorm1d(middle)
        self.bn2 = torch.nn.BatchNorm1d(middle)
        self.bn3 = torch.nn.BatchNorm1d(middle)

        self.bn4 = torch.nn.BatchNorm1d(fc)
        self.bn5 = torch.nn.BatchNorm1d(fc)
        self.dropout = torch.nn.Dropout(0.6)
        self.leakyrelu = torch.nn.LeakyReLU()

        self.attention = torch.nn.Linear(middle, 1)

    def graph_attention_pool(self, x, batch, edge_index):
        # 计算注意力分数
        attention_scores = self.attention(x)

        attention_scores = softmax(attention_scores, batch)
        # 加权汇总节点特征
        pooled_features = scatter_add(x * attention_scores, batch, dim=0)
        return pooled_features


    def forward(self, x, edge_index, edge_attr, batch):
        # center_node_feature = x[0]  # 第一个节点作为中心节点
        # x = torch.cat([center_node_feature.unsqueeze(0), x], dim=0)  # 拼接中心节点特征
        # main_node_index = 0
        # main_node_feature = x[main_node_index]
        x = self.bn(x)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.leakyrelu(x)
        x = self.bn1(x)

        x = self.conv2(x, edge_index, edge_attr)
        # x = x.relu()
        x = self.leakyrelu(x)
        x = self.bn2(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.leakyrelu(x)
        x = self.bn3(x)
        x = self.graph_attention_pool(x, batch, edge_index)
        # x = global_add_pool(x, batch )
        # x = global_max_pool(x, batch )  #
        # x = global_mean_pool(x, batch )

        x =self.bn0(x)
        # x = self.dropout(x)

        x = self.fc1(x)
        # x = torch.relu(x)
        x = self.leakyrelu(x)
        x =self.bn4(x)

        x = self.fc2(x)
        x = self.leakyrelu(x)
        x = self.bn5(x)

        x = self.fc3(x)
        # x = torch.sigmoid(x)
        # x = torch.softmax(x, dim=1)
        return x

from torch_geometric.nn import global_add_pool, GCNConv, GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, head=4, fc = 40):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=head )  #
        self.conv2 = GATConv(hidden_channels * head, hidden_channels, heads=head)  #
        self.conv3 = GATConv(hidden_channels * head, hidden_channels, heads=head)  #

        self.fc1 = torch.nn.Linear(hidden_channels* head, fc)
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

        # x = global_add_pool(x, batch=batch)
        x = self.bn3(x)

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


# 实例化模型
# model = ToxicityClassifier(in_channels=16, hidden_channels=160 ,middle= 160)
model = GAT( in_channels = 16, hidden_channels= 32, head=4, fc =48)
model = model.to(device)

# 加载模型参数
# model.load_state_dict(torch.load('./model/无增强数据 之链接中心  4  32 45 sig Algorithm 308+199.869.pth'))
model.load_state_dict(torch.load('./model/Best 418+199.914.pth'))
# model.load_state_dict(torch.load('./model/Best fisheye 136+198.793.pth'))
model.eval()
gama =0   ## -0.5 ~ 0.5





# data = []
# datalens = []
# for file in files:
#     file_path = os.path.join(path, file)
#     dataset = DiabetesDataset(file_path)
#     print(file, len(dataset))
#     datalens.append(len(dataset))
#     data.extend(dataset.data)


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    TP = 0
    TPFP = 0
    TPFN = 0
    runtime = 0
    with torch.no_grad():
        start_time = time.time()

        for data in data_loader:                   # only one

            batch = Batch.from_data_list(data)
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            pred = torch.round(out+gama)
            # pred = torch.argmax(out, dim=1)  # get the prediction result
            # batch.y = batch.y.squeeze().long()  # convert target label to class index
            correct += int(torch.eq(pred.view(-1), batch.y.view(-1)).sum())
            TPFP += int((torch.abs(1 - pred.view(-1)) < 1e-1).sum())
            # 注意张量尺寸

            TP += int((torch.logical_and(torch.eq(pred.view(-1), 1)
                                         , torch.eq(batch.y.view(-1), 1))).sum())

            TPFN += int( (torch.eq(batch.y.view(-1) ,1 )).sum() )
    runtime = time.time() - start_time




    # acc = correct / total
    precision = TP / TPFP if TPFP != 0 else 0.0
    recall = TP / TPFN
    print(f"Precision: {precision:.6f}, Recall: {recall:.6f}")

    return  precision ,recall,runtime ,pred

# path = './low/'
# path = './split200all/'
# path = './fisheyeall - copy/41/'
# path = './pool200all/'
path = './hanyang2002/'
files = [f for f in os.listdir(path) if f.endswith('.csv')]







import csv
import time
import re
def extract_png_names(filename ,endwith = '.jpg'):
    pattern = r"(\w+\.jpg)"

    if endwith  == '.png' :pattern = r"(\w+\.png)"
    # pattern = r"(\w+\.jpg)"
    png_names = re.findall(pattern, filename)
    png_names = [name.replace('from', '') for name in png_names]
    return png_names

def readname (th , file_name , endswith):
    if th == 1:
        match = re.search(r"from(\d+)_", file_name)
        k = match.group(1)
        print(k)
        image_path1 = k + 'l' + '.png'
        image_path2 = k + 'r' + '.png'
    else:
        [image_path1,image_path2] = extract_png_names(file_name, endwith= endswith)
         # = extract_png_names(file_name)[1]
    return image_path1,image_path2

def draw (path,filename , pred  ):
    image_path1, image_path2 = readname(1,filename , endswith = '.png')
    # image_path1, image_path2 = readname(0,filename , endswith = '.png')
    image1 = cv2.imread(path+ image_path1)
    image2 = cv2.imread(path+ image_path2)
    # cv2.imshow("",image1)
    # cv2.waitKey(0)
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]


    hmax = max(height1, height2)
    wmax = max(width1, width2)

    merged_width = width1 + width2
    merged_height = max(height1, height2)
    shift = 10
    # 创建一个新的空白图像作为拼接结果
    merged_image = np.zeros((merged_height, merged_width +shift, 3), dtype=np.uint8) + 255
    merged_image[:height1, :width1] = image1
    merged_image[:height2, width1+shift:] = image2

    whiteimage = np.zeros((hmax, wmax, 3), dtype=np.uint8) +255





    xy = np.loadtxt(path+ filename, delimiter=',', dtype=np.float32)
    labels = xy[::31, 0]  # 假设每隔31行取一个label
    pred = pred.cpu().numpy()  # 将预测值从torch张量转换为numpy数组

    labels = labels.astype(int)
    pred = pred.astype(int)


    results = []

    # 对每个样本进行判断
    for p, l in zip(pred, labels):
        if p == 1 and l == 1:
            results.append('TP')
        elif p == 1 and l == 0:
            results.append('FP')
        elif p == 0 and l == 1:
            results.append('FN')
        else:  # p == 0 and l == 0
            results.append('TN')

    # 输出结果
    # print(len(labels) , len(pred))
    # for i, result in enumerate(results):
    #     print(f"Sample {i + 1}: {result}")
    x = xy[::31, [3,4]]  # 假设每隔31行取一个label
    y = xy[::31, [5,6]]  # 假设每隔31行取一个label
    width1 += shift
    for i in range( len(results)):

        if results[i] == 'TN':
            pt1 = (int(x[i][0] * wmax),          int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_= (int(y[i][0] * wmax),          int(y[i][1] * hmax))
            # cv2.line(merged_image, pt1, pt2,  (0, 0, 0), 1)  # 绘制第一条线段，颜色为绿色，线宽为2
            cv2.line(whiteimage,   pt1, pt2_, (0, 0, 0), 1)  # 绘制第一条线段，颜色为绿色，线宽为2

    for i in range( len(results)):
        if results[i] == 'TP':
            pt1 = (int(x[i][0] * wmax),          int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_= (int(y[i][0] * wmax),          int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2,  (255, 0, 0), 1)  # 绘制第一条线段，颜色为绿色，线宽为2
            cv2.line(whiteimage,   pt1, pt2_, (255, 0, 0), 1)  # 绘制第一条线段，颜色为绿色，线宽为2
    for i in range( len(results)):
        if results[i] == 'FP':
            pt1 = (int(x[i][0] * wmax),          int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_= (int(y[i][0] * wmax),          int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2,  (0, 0, 255), 2)  # 绘制第一条线段，颜色为绿色，线宽为2
            cv2.line(whiteimage,   pt1, pt2_, (0, 0, 255), 2)  # 绘制第一条线段，颜色为绿色，线宽为2
        if results[i] == 'FN':
            pt1 = (int(x[i][0] * wmax),          int(x[i][1] * hmax))
            pt2 = (int(y[i][0] * wmax) + width1, int(y[i][1] * hmax))
            pt2_= (int(y[i][0] * wmax),          int(y[i][1] * hmax))
            cv2.line(merged_image, pt1, pt2,  (0, 255, 0), 2)  # 绘制第一条线段，颜色为绿色，线宽为2
            cv2.line(whiteimage,   pt1, pt2_, (0, 255, 0), 2)  # 绘制第一条线段，颜色为绿色，线宽为2
    #
    cv2.imshow('whiteimage', whiteimage)
    cv2.imshow('merged_image', merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./view/'+image_path1+ image_path2+'whiteimage.png',whiteimage)
    cv2.imwrite('./view/'+image_path1+ image_path2+'merged_image.png',merged_image)






def evaluate_and_write_to_csv(model, data_loader, csv_writer ,file_name):

    precision, recall ,runtime , pred = evaluate(model, data_loader)
    draw(path , file_name , pred)


    csv_writer.writerow([file_name,precision, recall, runtime])


with open('evaluation_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename','Precision', 'Recall', 'Runtime'])
    for file_name in files:
        data = DiabetesDataset(path+file_name)

        test_loader = DataListLoader(data, batch_size=len(data), shuffle=False)
        evaluate_and_write_to_csv(model, test_loader, writer ,file_name)
