import torch

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layer = list()
        for embed_dim in embed_dims:
            layer.append(torch.nn.Linear(input_dim, embed_dim))
            layer.append(torch.nn.BatchNorm1d(embed_dim))
            layer.append(torch.nn.ReLU())
            layer.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layer.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layer)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)