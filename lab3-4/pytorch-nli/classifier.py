import torch
import torch.nn as nn


class FCClassifier(nn.Module):
    """
    NLI classifier class, taking in concatenated premise and hypothesis
    embeddings as an input and returning the answer
    """
    def __init__(self, config):
        super(FCClassifier, self).__init__()
        self.config = config
        self.dropout = config.dropout
        if config.activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif config.activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        self.seq_in_size = 4*config.hidden_dim
        self.fc_dim = config.fc_dim
        self.out_dim = config.out_dim
        if self.config.encoder_type == 'BiLSTMEncoder':
            self.seq_in_size *= 2
        elif self.config.encoder_type == 'AttnEncoder':
            self.seq_in_size *= 4
        elif self.config.encoder_type == 'HBMP' or self.config.encoder_type == 'ConvEncoder':
            self.seq_in_size *= 6
        self.mlp = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.seq_in_size, self.fc_dim),
            self.activation,
            nn.Dropout(p=self.dropout),
            nn.Linear(self.fc_dim, self.fc_dim),
            self.activation,
            #nn.Dropout(p=self.dropout),
            nn.Linear(self.fc_dim, self.out_dim))

    def forward(self, prem, hypo):
        features = torch.cat([prem, hypo, torch.abs(prem-hypo), prem*hypo], 1)
        #print(features.size())

        output = self.mlp(features)
        return output
