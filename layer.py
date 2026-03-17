import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, conv_out_channels=16, kernel_size=3):
        super().__init__()
        layer = list()

        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=conv_out_channels,
                               kernel_size=kernel_size,
                               padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2)

        # 更新 input_dim 为卷积后的输出维度
        input_dim = conv_out_channels * (input_dim // 2)  # 假设pool层做了降维

        # 添加全连接层
        for embed_dim in embed_dims:
            layer.append(nn.Linear(input_dim, embed_dim))
            layer.append(nn.BatchNorm1d(embed_dim))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(p=dropout))
            input_dim = embed_dim

        # 添加输出层
        if output_layer:
            layer.append(nn.Linear(input_dim, 1))

        # 将层存储为 Sequential 模块
        self.mlp = nn.Sequential(*layer)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (batch_size, input_dim)
        """
        x = x.unsqueeze(1)

        # 添加卷积层和池化层
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten 操作，将卷积层的输出展平，送入全连接层
        x = x.view(x.size(0), -1)

        # 通过全连接层
        return self.mlp(x)