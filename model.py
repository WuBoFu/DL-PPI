import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random

from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv,GCNConv
class Inception(torch.nn.Module):
    def __init__(self,in_channels):
        super(Inception,self).__init__()
        self.branch1x1=torch.nn.Conv1d(in_channels,out_channels=1,kernel_size=1,stride=1)

        
        self.branch1x3_1=torch.nn.Conv1d(in_channels,3,kernel_size=3)
        self.branch1x3_2=torch.nn.Conv1d(3,1,kernel_size=1)

        self.branch1x5_1=torch.nn.Conv1d(in_channels,5,kernel_size=5,padding=0)
        self.branch1x5_2=torch.nn.Conv1d(5,1,kernel_size=1)
        self.branch_pool=torch.nn.Conv1d(in_channels,out_channels=1,kernel_size=1)
    def forward(self,x):
        branch1x1=self.branch1x1(x)

        branch1x3=self.branch1x3_1(x)
        branch1x3=self.branch1x3_2(branch1x3)

        branch1x5=self.branch1x5_1(x)
        branch1x5=self.branch1x5_2(branch1x5)

        branch_pool=F.avg_pool1d(x,1,1,0)

        branch_pool=self.branch_pool(branch_pool)

        outputs=[branch_pool,branch1x1,branch1x3,branch1x5]
        return torch.cat(outputs,dim=0)
class Attention(torch.nn.Module):
    def __init__(self,x):
        super(Attention,self).__init__()
        self.x=x
    def forward(self,x):
         # q=torch.tensor(q,dtype=torch.float32)
        a=torch.matmul(x.T,x)#q,k计算注意力
        b=F.softmax(a,dim=0)#注意力归一化
        c=torch.matmul(x,b)#注意力和v乘积。
        return c
Atten=Attention(torch.nn.Module)
class TenorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(16,16,32))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(32,2 * 16))
        self.bias = torch.nn.Parameter(torch.Tensor(32, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        embedding_1=Atten(embedding_1)
        embedding_2=Atten(embedding_2)

        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(16, -1))
        scoring = scoring.view(16, 32)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias) 



        return scores
NTN=TenorNetworkModule(torch.nn.Module)


class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, 
                hidden=512, use_jk=False, pool_size=3, cnn_hidden=1, train_eps=True, 
                feature_fusion= NTN, class_num=7):
        super(GIN_Net2, self).__init__()
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion

        
        self.incep1=Inception(in_feature)


        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=cnn_hidden, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_len / pool_size), gin_in_feature)
      
        self.gin_conv1 = GINConv( 
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=self.train_eps
                )
            )
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers*hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)
    
    def reset_parameters(self):
        
        self.conv1d.reset_parameters()
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()
        
        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.fc2.reset_parameters()
    

    def forward(self, x,edge_index, train_edge_id, p=0.5):
        
        x = x.transpose(1, 2)
        bn = nn.BatchNorm1d(1)
        x = self.conv1d(x)
       # x = self.incep1(x)
        x = self.bn1(x)
        x = self.maxpool1d(x)
        x = x.transpose(1, 2)
        x = self.global_avgpool1d(x)
        x = x.squeeze()

        x = self.fc1(x)

    

        # gin
        x = self.gin_conv1(x, edge_index)
        
        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs *= [x] # Agg层：融合

        if self.use_jk:
            x = self.jump(xs)
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin2(x) 


        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
    
        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1) 
        elif self.feature_fusion == 'NTN':
            x = NTN(x1,x2)
        else:
            x = torch.mul(x1, x2) 
        x = self.fc2(x) 

        return x