import datetime
import errno
import os
import pickle
import random
from pprint import pprint
import random
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from scipy import io as sio, sparse
from sklearn.model_selection import KFold

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

# 计算节点间存在连接可能性的得分。
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
        
def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def construct_neg_graph(G, k, edge_type):
    src_type, e_type, dst_type = edge_type
    n_src = G.num_nodes(src_type)
    n_dst = G.num_nodes(dst_type)
    src_G, dst_G = G.edges(etype = edge_type)
    mask_tmp = [set() for _ in range(n_src)]
    num_of_pos_edges = len(src_G)
    for i in range(num_of_pos_edges):
        mask_tmp[src_G[i].item()].add(dst_G[i].item())
    eids = np.arange(G.num_edges(etype=edge_type))
    neg_G = dgl.remove_edges(G, eids, etype=edge_type)
    for i in range(n_src):
        if len(mask_tmp[i]) == 0:
            continue
        v2 = mask_tmp[i]
        neg_dst = set()
        while len(neg_dst) < k * len(v2):
            if len(neg_dst) + len(v2) >= n_dst:
                break
            x = torch.randint(0, n_dst, (1, )).item()
            if x not in v2:
                neg_dst.add(x)
        for y in neg_dst:
            neg_G.add_edges(i, y, etype=edge_type)
    return neg_G

# def generate_traning_batch(postivefile,negtivefile):
#     with open(postivefile,'rb')as f:
#         now_pos = pickle.load(f)
#     with open(negtivefile,'rb')as f:
#         now_neg = pickle.load(f)
#     all_nodes = {}
#     check = set()
#     for id_ in now_pos:
#         all_nodes[id_] = 1
#     for id_ in now_neg:
#         all_nodes[id_] = 0
#     all_ids  = list(all_nodes.keys())
#     test_size = int(0.1*len(all_ids))
#     test_ids = random.sample(all_ids,test_size)
#     test_labels = [all_nodes[id] for id in test_ids]
#     test_ids_tensor = torch.tensor(test_ids)
#     test_labels_tensor = torch.tensor(test_labels)
#     train_set = list(set(all_ids)-set(test_ids))
#     train_table = {}
#     for idx,train_id in enumerate(train_set):
#         train_table[idx] = train_id
#     train_batchs=[]
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#     for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(train_set)):
#         print(train_idx)
#         x_train = [train_table[idx] for idx in train_idx]
#         y_train = [all_nodes[x] for x in x_train]
#         x_valid = [train_table[idx] for idx in valid_idx]
#         y_valid = [all_nodes[x]for x in x_valid]
#         train_batchs.append((torch.tensor(x_train),torch.tensor(x_valid),torch.tensor(y_train),torch.tensor(y_valid)))
#     return  train_batchs,test_ids_tensor,test_labels_tensor

def generate_traning_batch(postivefile,negtivefile):
    '''
        Five-fold cross-validation is used to generate the training, validation, and test sets.
        Each time, the training and test sets are first split, and then the validation set is randomly sampled from the training set, 
        ensuring a 3:1:1 ratio between the training, validation, and test sets, respectively. 
        Moreover, we ensure that the test and validation sets do not appear in the training set.
        
    '''
    with open(postivefile,'rb')as f:
        now_pos = pickle.load(f)
    with open(negtivefile,'rb')as f:
        now_neg = pickle.load(f)
    all_nodes = {}
    check = set()
    for id_ in now_pos:
        all_nodes[id_] = 1
    for id_ in now_neg:
        all_nodes[id_] = 0
    
        
#     all_ids1  = list(all_nodes.keys())
#     all_ids = list(set(all_ids1)-numb)
    all_ids = list(all_nodes.keys())
    test_size = int(0.1*len(all_ids))
    test_ids = random.sample(all_ids,test_size)
    test_labels = [all_nodes[id] for id in test_ids]
    test_ids_tensor = torch.tensor(test_ids)
    test_labels_tensor = torch.tensor(test_labels)
    train_ids = list(set(all_ids)-set(test_ids))
    random.shuffle(train_ids)
    fold_size = len(train_ids) // 5
    folds = [train_ids[i:i+fold_size] for i in range(0, len(train_ids), fold_size)]
    train_batchs=[]
    cnt = 0
    for i, fold in enumerate(folds):
        # 第i个fold作为测试集，其余作为训练集
        val_ids = fold
        t_train_ids = [id_ for id_ in train_ids if id_ not in val_ids]

        # 根据训练、和验证集ID，从all_nodes中获取节点标签
        train_labels = [all_nodes[id] for id in t_train_ids]
        val_labels = [all_nodes[id] for id in val_ids]
        train_ids_tensor = torch.tensor(t_train_ids)
        val_ids_tensor = torch.tensor(val_ids)
        train_labels_tensor = torch.tensor(train_labels)
        val_labels_tensor = torch.tensor(val_labels)
        train_batchs.append((train_ids_tensor,val_ids_tensor,
                         train_labels_tensor,val_labels_tensor))
        cnt+=1
        if(cnt==5):
            break
    return train_batchs,test_ids_tensor,test_labels_tensor


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args["log_dir"], "{}_{}".format(args["dataset"], date_postfix)
    )

    if sampling:
        log_dir = log_dir + "_sampling"

    mkdir_p(log_dir)
    return log_dir


# The configuration below is from the paper.
default_configure = {
    "lr": 0.005,  # Learning rate
    "num_heads": [8],  # Number of attention heads for node-level attention
    "hidden_units": 8,
    "dropout": 0.6,
    "weight_decay": 0.001,
    "num_epochs": 200,
    "patience": 100,
}

sampling_configure = {"batch_size": 20}


def setup(args):
    args.update(default_configure)
    set_random_seed(args["seed"])
    args["dataset"] = "ACMRaw" if args["hetero"] else "ACM"
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    args["log_dir"] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    args["log_dir"] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()





class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
