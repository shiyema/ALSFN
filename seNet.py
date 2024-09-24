import torch
import torch.nn as nn
from gumbel_softmax import gumbel_softmax, rao_gumbel, st_gumbel_softmax


class switchNet(nn.Module):
    def __init__(self, input_size, num_switchs):
        super(switchNet, self).__init__()
        self.B = input_size[0]
        self.C = input_size[1]
        self.L = input_size[2]
        self.hidden_figure_dimension = 64
        self.hidden_figure_dimension_v = 40
        self.dec_q = nn.Linear(input_size[2], self.hidden_figure_dimension)
        self.dec_k = nn.Linear(input_size[2], self.hidden_figure_dimension)
        self.dec_v = nn.Linear(self.L, self.hidden_figure_dimension_v)
        self.softmax = nn.Softmax(dim=-1)
        self.classifier = nn.Sequential(  # 定义分类网络结构
            nn.Dropout(p=0.05),  # 减少过拟合
            nn.ReLU(True),
            nn.Linear(15360, num_switchs)
        )

    def forward(self, x, policy=None):
        X_encoded = x
        proj_query = self.dec_q(X_encoded)  # proj_query.shape =
        proj_key_temp = self.dec_k(X_encoded)  # proj_key_temp =
        proj_value_temp = self.dec_v(X_encoded)  # proj_value_temp =
        energy = torch.bmm(proj_query, proj_key_temp.permute(0, 2, 1))  # energy.shape =
        temp_attention = self.softmax(energy)  # temp_attention.shape =
        context_vector = torch.bmm(proj_value_temp.permute(0, 2, 1),
                                   temp_attention.permute(0, 2, 1))  # context_vector.shape =
        out = self.classifier(context_vector.reshape(context_vector.shape[0], context_vector.shape[1]*context_vector.shape[2]))
        return out

class SENET(nn.Module):  # 回归网络
    def __init__(self):
        super(SENET, self).__init__()
        self.switch_model = switchNet([16, 384, 1024], 3)
        # self.switch_model = PairswitchNet([args.batchsize, 3, 200, 3], args, 10)
        self.device = 0


    def forward(self, x):
        device = torch.device("cuda:{}".format(self.device) if torch.cuda.is_available() else "cpu")
        probs = self.switch_model(x.float())
        probs = probs / probs.sum(dim=1, keepdim=True)
        #policy = gumbel_softmax(probs.view(probs.size(0), -1, 2), device)[:, :, 1]
        #policy = gumbel_softmax(probs, device)
        #policy = st_gumbel_softmax(probs, device)
        policy = rao_gumbel(probs, device)
        #a = [[4], [5], [6]]
        #a = torch.tensor(a).to(device)
        #out = policy @ a.float()

        # if out.shape[0] == 1:
        #     print(policy)
        return policy
