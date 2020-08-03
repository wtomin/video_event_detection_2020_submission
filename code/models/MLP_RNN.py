import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_units, dropout=0.3):
        super(MLP, self).__init__()
        input_feature_dim = hidden_units[0]
        num_layers = len(hidden_units)-1
        assert num_layers>0
        assert hidden_units[-1]==256
        fc_list = []
        for hidden_dim in hidden_units[1:]:
            fc_list += [ nn.Dropout(dropout),
                        nn.Linear(input_feature_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True)
                        ]
            input_feature_dim = hidden_dim
        self.mlp = nn.Sequential(*fc_list)
    def forward(self, input_tensor):
        bs, num_frames, feature_dim = input_tensor.size()
        input_tensor = input_tensor.view(bs*num_frames, feature_dim)
        out = self.mlp(input_tensor)
        return out.view(bs, num_frames, -1)
        
class MLP_RNN(nn.Module):
    def __init__(self, mlp_hidden_units, num_classes):
        super(MLP_RNN, self).__init__()
        self.mlp = MLP(mlp_hidden_units)
        feature_dim = mlp_hidden_units[-1]
        self.rnns = nn.ModuleList([nn.GRU(feature_dim, feature_dim//2, bidirectional=True),
                    nn.GRU(feature_dim, feature_dim//2,  bidirectional=True)])
        self.classifier = nn.Sequential(nn.Dropout(0.3),
                                        nn.Linear(feature_dim, num_classes))
    def forward(self, data):
        bs, num_frames = data.size(0), data.size(1)
        features_cnn = self.mlp(data)
        outputs_rnns = features_cnn
        for rnn_layer in self.rnns:
            outputs_rnns, _ = rnn_layer(outputs_rnns)
            outputs_rnns = F.relu(outputs_rnns)
        outputs_rnns = outputs_rnns.view(bs* num_frames, -1)
        out = self.classifier(outputs_rnns)
        out = out.view(bs, num_frames, -1)
        return out