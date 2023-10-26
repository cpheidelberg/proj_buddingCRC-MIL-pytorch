
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class Aggregation(nn.Module):
    def __init__(self, linear_nodes = 15, attention_nodes = 15, dim = 0, aggregation_func = None):
        super().__init__()
        self.linear_nodes = linear_nodes
        self.attention_nodes = attention_nodes
        self.dim = dim
        self.aggregation_func = aggregation_func

        self.attention_layer = nn.Sequential(
            nn.Linear(self.linear_nodes, self.attention_nodes),
            nn.Tanh(),
            nn.Linear(self.attention_nodes,1)
        )

    def forward(self, x, dim = None):
        gate = self.attention_layer(x)
        #print(gate)
        attention_map= x*gate
        if dim is None:
            dim = self.dim

        if self.aggregation_func is None:
            attention = torch.mean(attention_map, dim = dim)
        else:
            attention = self.aggregation_func(attention_map, dim=dim)

        return attention

#%%
#agg_func = Aggregation(dim=0)
#m = agg_func(output_prepNN)
#print(m)
#print(m.size())