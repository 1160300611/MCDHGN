import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from dgl.nn import GraphConv
from dgl.nn import GCN2Conv
from dgl.sampling import RandomWalkNeighborSampler


class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, activation, feat_drop_rate, attn_drop_rate):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.feat_drop = nn.Dropout(feat_drop_rate)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.activation = activation

        for i in range(num_heads):
            self.heads.append(self.build_head())

        self.weight = nn.Parameter(torch.Tensor(in_feats, num_heads * out_feats))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)

    def build_head(self):
        return nn.Linear(self.in_feats, self.out_feats, bias=False)

    def forward(self, graph, inputs):
        h = self.feat_drop(inputs)
        outputs = []
        for i, attention_head in enumerate(self.heads):
            # compute attention coefficients
            a = self.attn_drop(attention_head(h))

            # compute softmax
            graph.ndata['attn'] = F.softmax(torch.sum(a * h, dim=-1) / (self.out_feats ** 0.5), dim=0)
            # compute the linear combination of neighborhood features
            h_prime = dgl.prop_nodes_topo(graph, message_func=dgl.function.u_mul_e('h', 'attn'),
                                          reduce_func=dgl.function.sum('m', 'neigh'))
            outputs.append(h_prime)

        h = torch.cat(outputs, dim=-1)
        h = torch.matmul(h, self.weight)

        return self.activation(h)
    
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1),beta[0]  # (N, D * K)


class MySampler(object):
    def __init__(self, g, metapath_list, num_neighbors):
        self.sampler_list = []
        for metapath in metapath_list:
            # note: random walk may get same route(same edge), which will be removed in the sampled graph.
            # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
            self.sampler_list.append(
                RandomWalkNeighborSampler(
                    G=g,
                    num_traversals=1,
                    termination_prob=0,
                    num_random_walks=num_neighbors,
                    num_neighbors=num_neighbors,
                    metapath=metapath,
                )
            )

    def sample_blocks(self, seeds):
        block_list = []
        for sampler in self.sampler_list:
            frontier = sampler(seeds)
            # add self loop
            frontier = dgl.remove_self_loop(frontier)
            frontier.add_edges(seeds.clone().detach() , seeds.clone().detach() )
            block = dgl.to_block(frontier, seeds)
            block_list.append(block)

        return seeds, block_list


class linear_module(nn.Module):
    def __init__(self, dim):
        super(linear_module, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim/2), bias=True)
        self.fc2 = nn.Linear(int(dim/2), 2, bias=False)
        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, x, x2=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class MyLayer(nn.Module):
    """
    Model layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(MyLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        gat_attention_weights = [] 
        for i, g in enumerate(gs):
            embeddings,alpha_weight = self.gat_layers[i](g,h,get_attention=True)
            semantic_embeddings.append(embeddings.flatten(1))
            gat_attention_weights.append(alpha_weight)
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        gat_out,beta = self.semantic_attention(semantic_embeddings)
        return gat_out,gat_attention_weights,beta  # (N, D * K)


class MCDHGN(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(MCDHGN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            MyLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                MyLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        dim = hidden_size*num_heads[-1]
        
 
        self.predict = linear_module(dim=dim)
        #self.predict=nn.Linear(hidden_size * num_heads[-1], out_size)
        #self.conv1 = GraphConv(dim, 128, norm='both', weight=True, bias=True)
        #self.conv2 = GraphConv(128,2)
        #self.conv = GCN2Conv(dim,layer=1,alpha = 0.8,project_initial_features=True, allow_zero_in_degree=True)
    def forward(self, g, h):
        for i, gnn in enumerate(self.layers):
            if i ==0:
                h,alpha,beta = gnn(g,h)
            else:
                h,_,_ = gnn(g,h)
        
        return h,self.predict(h),alpha,beta
