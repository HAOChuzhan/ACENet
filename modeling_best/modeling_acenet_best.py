from torch_geometric.nn.inits import glorot, zeros
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add, scatter, scatter_mean, scatter_sum
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import MessagePassing
from torch.autograd import Variable
from modeling_best.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils_best.data_utils import *
from utils_best.layers import *
import torch.nn.functional as F
import numpy as np
import pdb


class ACENet_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim, input_size, hidden_size, output_size,
                 dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.k = k
        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size//2)

        self.basis_f = 'sin'  # ['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size//2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size//2)
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size), torch.nn.BatchNorm1d(
            hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))

        # 新增的一些参数层
        self.alpha = 0.8  # (1)0.8
        # self.q_weight_mlp = nn.Linear(2*hidden_size, 1)
        # self.q_weight_mlp = torch.nn.Sequential(torch.nn.Linear(2*hidden_size, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, 1))
        self.MHAtt = SelfAttLayer(2, 2, hidden_size, hidden_size, 5)
        # self.k_att = nn.Linear(hidden_size, 1)
        # TEMP = self.alpha*(1-self.alpha)**np.arange(self.k+1)
        '''
        TEMP公式
        '''
        # TEMP = self.alpha*(1-self.alpha)**np.arange(self.k+1)
        # self.temp = nn.Parameter(torch.tensor(TEMP))

        # hidden_size=concept_dim=200
        self.gnn_layers = nn.ModuleList([GATConvE(
            args, sent_dim, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])

        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    # 剪枝掉无用的结点
    def pruning(self, _X, edge_index, edge_type, _node_type, _node_feature_extra, context_node_list, k):
        # context_node_emb = _X[:, 0, :] # 获取第一个的context的结点表示，然后根据其余结点与该节点的相似度来排序 [bs, 1, emd_dim]
        # IndexError: too many indices for tensor of dimension 2
        # 这里先进行图之间的多头注意力增强子图之间的联系 _X[bs*node, hidden_dim] _node_type[bs*node, ]
        c_entity_index = (_node_type == 3)
        c_entity_index = torch.tensor([i for i, x in enumerate(c_entity_index) if x]).to(
            _X.device)  # [0, 200, 400 ... 1800]
        # [context_nums, hidden_size] [10, 200] 一个mini_bs有两个子图
        c_entity_emb = _X.index_select(0, c_entity_index)

        context_node_emb, gnn_att = self.MHAtt(
            c_entity_emb, c_entity_emb)  # 参数共享在几层的gnn中
        # print(f'GNNs中间层{k+1}的Att矩阵:{gnn_att}')
        # pdb.set_trace()
        # 修改原先的图中的q-a pair的node_emb:
        for i in c_entity_index:
            _X[i, :] = context_node_emb[i//self.hidden_size]

        # 置于_X前还是后，需进行实验
        context_node_list.append(_X.index_select(0, c_entity_index))

        _X = self.activation(_X)
        _X = F.dropout(_X, self.dropout_rate, training=self.training)

        return _X, edge_index, edge_type, _node_type, _node_feature_extra, context_node_list

    # 这里的5个子图之间的context节点可以进行结合交互
    def mp_helper(self, sent_vecs, _X, edge_index, edge_type, edge_batch, _node_type, _node_feature_extra):
        context_node_list = []  # 保存中间的5个q-a节点信息
        # hidden = _X*(self.temp[0])
        for k in range(self.k):
            # , edge_index, edge_type, _node_type, _node_feature_extra
            _X = self.gnn_layers[k](
                sent_vecs, _X, edge_index, edge_type, edge_batch, _node_type, _node_feature_extra)
            _X, edge_index, edge_type, _node_type, _node_feature_extra, context_node_list = self.pruning(
                _X, edge_index, edge_type, _node_type, _node_feature_extra, context_node_list, k)
            # hidden = hidden + self.temp[k+1]*_X

        context_node_h = torch.stack(
            context_node_list, dim=1)  # [bs(10), k, hidden]

        return _X, context_node_h

    def forward(self, sent_vecs, H, A, node_type, node_score, cache_output=False):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type, edge_batch)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        # Embed type
        T = make_one_hot(node_type.view(-1).contiguous(),
                         self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        # [batch_size, n_node, dim/2]
        node_type_emb = self.activation(self.emb_node_type(T))

        # Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(
                0).float().to(node_type.device)  # [1,1,dim/2]
            js = torch.pow(1.1, js)  # [1,1,dim/2]
            B = torch.sin(js * node_score)  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(
                self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(
                self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            # [batch_size, n_node, dim/2]
            B = self.activation(self.B_lin(node_score))
            node_score_emb = self.activation(
                self.emb_score(B))  # [batch_size, n_node, dim/2]

        # previous layer [bs, n_node, node_dim]
        X = H
        # edge features
        # edge_index: [2, total_E]   edge_type: [total_E, ]  edge_batch: [total_E, ] where total_E is for the batched graph
        edge_index, edge_type, edge_batch = A
        # [bs*node_n, node_dim]
        bs = X.size(0)
        node_n = X.size(1)
        # [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node 1
        _X = X.view(-1, X.size(2)).contiguous()
        # [bs*n_node]
        # [`total_n_nodes`, ]    2
        _node_type = node_type.view(-1).contiguous()
        # [bs, n_node, dim/2 * 2] -> [bs*n_node, dim/2 * 2]
        # _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim] 2
        _node_feature_extra = (node_type_emb, node_score_emb, bs, node_n)
        # sent_vecs来保证我们剪枝的时候不会太偏出问题的语义

        _X, context_node_h = self.mp_helper(
            sent_vecs, _X, edge_index, edge_type, edge_batch, _node_type, _node_feature_extra)

        # context_node_h --> [bs, k, hidden]
        # [batch_size, n_node, hidden_size]
        X = _X.view(node_type.size(0), node_type.size(1), -1)

        context_node_PLM = H[:, 0]  # [bs, hidden_dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        # output = self.activation(self.Vx(X))
        output = self.dropout(output)

        return output, context_node_h, context_node_PLM


# Decoder
class ACENet(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, relation_dim, sent_dim,
                 n_concept, concept_dim, concept_in_dim, mini_bs, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, pretrained_relation_emb=None, freeze_ent_emb=True,
                 init_range=0.02):
        super().__init__()
        self.init_range = init_range

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)  # sentencer2node dim

        # 新增模型参数层
        # self.svec2nvec2 = nn.Linear(sent_dim, concept_dim) # sentencer2node dim
        self.attention = SelfAttLayer(
            mini_bs, n_attention_head, sent_dim, sent_dim, 5)
        # self.ntype_tansform = TypedLinear(concept_dim, concept_dim, n_etype)
        # self.k_att = nn.Linear(sent_dim, k)

        self.concept_dim = concept_dim
        self.edge_type = n_etype
        self.activation = GELU()

        self.gnn = ACENet_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype, sent_dim=sent_dim,
                                          input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim, dropout=p_gnn)

        self.pooler = MultiheadAttPoolLayer(
            n_attention_head, sent_dim, concept_dim)
        self.fc = MLP(sent_dim + 2*concept_dim, fc_dim, 1,
                      n_fc_layer, p_fc, layer_norm=True)
        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    '''
    1. 引入双线性融合
    2. 引入GPO来改进GCN的一个消息传播
    3. 如何筛选出与问题的context最为相关的一些结点来进行推理，这里使用soft-prune多头机制
    
    '''

    def forward(self, attention_mask, sent_vecs, last_hidden_states, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None, cache_output=False):
        """
        sent_vecs: (batch_size, dim_sent) 
        last_hidden_states: (batch_size, seq_len, dim_sent)
        concept_ids: (batch_size, n_node) 
        adj: edge_index, edge_type, edge_batch
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """
        u = self.svec2nvec(sent_vecs)  # [,200]
        gnn_input0 = self.activation(u).unsqueeze(
            1)  # (mini_bs*n_c, 1, dim_node)
        # (mini_bs*n_c, n_node-1, dim_node) emb_data=None
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data)
        gnn_input1 = gnn_input1.to(node_type_ids.device)

        # 这里如何更好的融合两者之间的信息,再着就是如何排除部分的噪声结点 毕竟不是所有结点都是有用的
        # 将映射到dim_node空间的上下文(形成一个结点)与图谱concept节点编码拼接起来，形成一个完整的图节点
        # [batch_size, n_node, concept_dim]
        gnn_input = torch.cat([gnn_input0, gnn_input1], dim=1)
        '''
        这里是节点的初始化 
        
        ## 改GNN的node节点的初始化方式，加入了text的token表达来做cross attention
        h_query = self.svec2nvec2(last_hidden_states)  # [batch_size, seq_len, concept_dim]

        # 原始的节点表示，用TextEncoder编码的sent的表示来引导剪枝,先把sent与节点的编码进行交互，有助于之后的gnn中的推理；
        q_logit = torch.matmul(h_query, gnn_input.transpose(-1, -2)) # [batch_size, seq_len, n_node]
        q_logit = q_logit * attention_mask.float().unsqueeze(2).to(q_logit.device) # [batch_size, seq_len, n_node] * [bs, seq_len, 1]
        q_logit = q_logit / (torch.sum(q_logit, dim=1, keepdim=True) + 1e-6) # seq_len行归一化 -> [batch_size, seq_len, n_node]

        q_word_att = torch.sum(q_logit, dim=1, keepdim=True) # [batch_size, 1, n_node]
        q_word_att = torch.softmax(q_word_att, dim=2).transpose(-1, -2) # [batch_size, n_node, 1]

        gnn_input =  q_word_att * gnn_input # + gnn_input
        '''

        '''
        这里引入节点的权重node_att和边的类型权重(并不是所有边都是同等重要的)
        '''
        # gnn_input = self.ntype_tansform(gnn_input, node_type_ids) # 这里的4是指的节点的类型数 效果不好
        gnn_input = self.dropout_e(gnn_input)

        # edge
        # edge_index, edge_type = adj
        # edge_mask = edge_type < 34 # [E,]

        # Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) <
                 adj_lengths.unsqueeze(1)).float()  # 0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        # [batch_size, n_node, 1]
        node_scores = node_scores - node_scores[:, 0:1, :]
        node_scores = node_scores.squeeze(2)  # [batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm = (torch.abs(node_scores)).sum(
            dim=1) / adj_lengths  # [batch_size, ]
        node_scores = node_scores / \
            (mean_norm.unsqueeze(1) + 1e-05)  # [batch_size, n_node]
        node_scores = node_scores.unsqueeze(2)  # [batch_size, n_node, 1]

        # 这里加入一个sent_vec之间的交互
        sent_vecs_for_pooler, self_att_matrix = self.attention(
            sent_vecs, sent_vecs)

        # print('初始的Att矩阵:',self_att_matrix)
        # pdb.set_trace()
        # GNN的输入信息包含有问题编码,节点嵌入编码，邻接矩阵，结点的类型id，节点的得分
        # 这里从sent_vecs->sent_vecs_for_att
        # gnn_output = self.gnn(sent_vecs, gnn_input, adj, node_type_ids, node_scores)

        # print(adj[0].size(), adj[1].size()) torch.Size([2, 4742]) torch.Size([4742])

        gnn_output, context_node_h, context_node_plm = self.gnn(
            u, gnn_input, adj, node_type_ids, node_scores)  # node_type_ids：[batch_size, n_node]
        # 最后一层和初始层分别过线性层后相加激活drop的context-node
        # (batch_size, concept_dim)  取出了一个bs中所有样本中的第一个node-context节点的emb
        Z_vecs = gnn_output[:, 0]

        '''这里是原始的gnn-att weights
        k_logit = self.k_att(sent_vecs).unsqueeze(2)
        # k_logit = self.k_att(sent_vecs).unsqueeze(2) # [bs, k, 1]
        k_att = torch.softmax(k_logit, dim=1)
        last_Z_vecs = torch.sum(context_node_h*k_att, dim=1) # [bs, k, hidden]*[bs, k, 1] --> [bs, hidden]
        '''

        mask = torch.arange(node_type_ids.size(
            1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1)  # 1 means masked out
        # pool over all KG nodes 得到一个与node_type==3长度相等的一个mask
        mask = mask | (node_type_ids == 3)
        # a temporary solution to avoid zero node 保证第一个context节点一定是未被mask的
        mask[mask.all(1), 0] = 0
        # sent_vecs_for_pooler = sent_vecs
        # multi-head attention for all node—type==0,1,2  node

        # 从 sent_vecs_for_pooler改成sent_vecs 重新改回去，这里主要是将我们的相关的evidence来与问题做多头
        # [bs, sent_dim] [bs, node, node_dim] [bs, node_id_len]->[b, n*d_v] [n*b, l]
        graph_vecs, pool_attn = self.pooler(
            sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        # 尝试加上最初的Node的表示
        # [b, n*d_v=concept_dim],[b, sent_dim],[b, gnn_dnn]列向拼接，col值相加，然后经过dropout
        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))
        # concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, last_Z_vecs, Z_vecs), 1)) # [b, n*d_v=concept_dim],[b, sent_dim],[b, gnn_dnn]列向拼接，col值相加，然后经过dropout

        logits = self.fc(concat)  # 然后经过前馈层来输出最后的logits
        return logits, pool_attn


class LM_ACENet(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype, relation_dim,
                 n_concept, concept_dim, concept_in_dim, mini_bs, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, pretrained_relation_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}):
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = ACENet(args, k, n_ntype, n_etype, relation_dim, self.encoder.sent_dim,
                              n_concept, concept_dim, concept_in_dim, mini_bs, n_attention_head,
                              fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                              pretrained_concept_emb=pretrained_concept_emb, pretrained_relation_emb=pretrained_relation_emb, freeze_ent_emb=freeze_ent_emb,
                              init_range=init_range)

    def forward(self, *inputs, layer_id=-1, cache_output=False, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        # Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [x.view(
            x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x, []) for x in inputs[-2:]]

        # lm_inputs = (example_ids, all_label, *data_tensors)
        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        '''
        print("concept_ids size:", concept_ids.size())
        print("node_type_ids size:", node_type_ids.size())
        print("node_scores size:", node_scores.size()) 
        print("adj_lengths size:", adj_lengths.size())
        
        concept_ids size: torch.Size([10, 200])
        node_type_ids size: torch.Size([10, 200])
        node_scores size: torch.Size([10, 200, 1])
        adj_lengths size: torch.Size([10])
       
        print("edge_index size:", len(edge_index))
        print(edge_index[0].size())
        print("edge_type size:", len(edge_type))  torch.Size([2, 38])
        print(edge_type[0].size())                torch.Size([38, ])
        edge_index size: 10
        edge_type size: 10
        
        # 10个子图的边数记录
        torch.Size([2, 38])
        torch.Size([2, 26])
        torch.Size([2, 176])
        torch.Size([2, 126])
        torch.Size([2, 8])
        torch.Size([2, 628])
        torch.Size([2, 896])
        torch.Size([2, 576])
        torch.Size([2, 594])
        torch.Size([2, 1674])
        '''

        edge_index, edge_type, edge_batch = self.batch_graph(
            edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device), edge_batch.to(
            node_type_ids.device))  # edge_index: [2, total_E]   edge_type: [total_E, ]

        # print("edge_index size:", edge_index.size()) edge_index size: torch.Size([2, 4742])

        # all_hidden_states 包含了初始的嵌入编码加模型每一层的输出all_hidden_states[-1]即为last_hidden_states =>[bs, seq_len, hidden_dim]
        sent_vecs, all_hidden_states, attention_mask = self.encoder(
            *lm_inputs, layer_id=layer_id)  # [mini-bs*n_c, 1024] + 13层[mini-bs*n_c, seq_len, 1024]
        last_hidden_states = all_hidden_states[-1]
        logits, attn = self.decoder(attention_mask, sent_vecs.to(node_type_ids.device), last_hidden_states.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    emb_data=None, cache_output=cache_output)
        logits = logits.view(bs, nc)
        if not detail:
            return logits, attn
        else:
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig
            # edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            # edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        # edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        # edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        # >>> edge_index2
        # tensor([[ 7,  0,  4,  8,  1, 13, 10, 11, 14, 11, 26, 27, 21, 28, 21], E = mini-bs*n_c*38(每个子图的边数是不一样的)
        #         [ 0,  3,  8,  3,  1, 13, 17, 19, 12, 13, 22, 28, 27, 25, 22]])
        # >>> edge_type
        # tensor([0, 2, 3, 6, 3, 4, 9, 6, 9, 3, 2, 2, 4, 8, 6])

        n_examples = len(edge_index_init)  # mini-bs*n_c = 10
        # 这里是把10个子图的边全部拼接起来了，每一个子图中的节点id都是递增了200
        edge_index = [edge_index_init[_i_] + _i_ *
                      n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        # [total_E, ] # 这里是直接把10个子图的所有边的类型直接横向拼接起来
        edge_type = torch.cat(edge_type_init, dim=0)

        # 添加上每个边属于哪个q-a pair中
        edge_batch = [
            [i]*edge_index_init[i].size(1) for i in range(n_examples)]
        edge_batch = torch.tensor([y for edge in edge_batch for y in edge])

        return edge_index, edge_type, edge_batch


class LM_ACENet_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        # 去对应的字典中索引预训练的LM 这里是model_name:roberta-large和type:roberta
        model_type = MODEL_NAME_TO_CLASS[model_name]
        print('train_statement_path', train_statement_path)

        model_name = '~/sdb/acenet/Pretrained/roberta-large'
        print('model name', model_name)
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(
            train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(
            dev_statement_path, model_type, model_name, max_seq_length)
        # *self.train_encoder_data = all_input_ids, all_input_mask, all_segment_ids, all_output_mask,
        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(
            train_adj_path, max_node_num, num_choice, args)

        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(
            dev_adj_path, max_node_num, num_choice, args)
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [
                   self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(
            0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(
                test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(
                test_adj_path, max_node_num, num_choice, args)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(
                0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        # 这里的inhouse的训练集是train在inhouse_id中，测试集是不在的inhouse集合之中的
        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])
            print("inhouse_train_indexes_len:",
                  self.inhouse_train_indexes.size(0))
            print("inhouse_test_indexes_len:",
                  self.inhouse_test_indexes.size(0))

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
                print("subsample_inhouse_train_indexes_len:",
                      self.inhouse_train_indexes.size(0))
                print("subsample_inhouse_test_indexes_len:",
                      self.inhouse_test_indexes.size(0))
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train]
                                           for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train]
                                           for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [
                           self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            # random train_indexers
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)


###############################################################################
############################### GNN architecture ##############################
###############################################################################


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """

    def __init__(self, args, sent_dim, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype
        self.n_etype = n_etype  # 4 and 38
        self.edge_encoder = edge_encoder

        # For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key2 = nn.Linear(2*emb_dim, head_count * self.dim_per_head)
        # self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(
            2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(
            emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.weight_mlp = torch.nn.Sequential(torch.nn.Linear(
            3*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1))
        # self.weight_mlp_ = MLP(3*emb_dim, emb_dim, 1, 1, 0.2, batch_norm=False, layer_norm=True)

        # sentvec_dim2hidden_dim
        # self.sent2hidden = nn.Linear(sent_dim, emb_dim)
        # self.matrixAtt = MatrixAttention()
        # self.linear1 = nn.Linear(emb_dim, emb_dim)
        # self.linear2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, sent_vecs, x, edge_index, edge_type, edge_batch, node_type, node_feature_extra, return_attention_weights=False):
        # sent_vecs: [,sent_dim]
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        # new
        # sent_vecs: [,sent_dim]
        # x: [bs, N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]
        # Prepare edge feature
        '''
        x = x.view(-1, x.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node 
        # [bs*n_node]
        node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]    2
        # [bs, n_node, dim/2 * 2] -> [bs*n_node, dim/2 * 2]
        node_feature_extra = node_feature_extra.view(node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim] 2
        '''
        '''
        print("sent_vecs size:", sent_vecs.size())
        print("edge_type size:", edge_type.size())
        print("x size:", x.size())
        print("edge_index size:", edge_index.size())
        print("node_type size:", node_type.size())
        print("node_feature_extra size:", node_feature_extra.size())
        sent_vecs size:          torch.Size([10, 1024])
        edge_type size:          torch.Size([8166])
        x size:                  torch.Size([2000, 200])
        edge_index size:         torch.Size([2, 8166])
        node_type size:          torch.Size([2000])
        node_feature_extra size: torch.Size([2000, 200])
        '''

        # [E, 39] torch.Size([6742, 39])
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)

        N = x.size(0)
        # print("N:", N) N: 2000
        self_edge_vec = torch.zeros(N, self.n_etype + 1).to(edge_vec.device)
        # [N, 39] 最后一列设置为1 这里是 the edge type set L is assumed to contain a special edge type ö for self-loops v -> 上边有一个自环的圆圈○ v, allowing state associated with a node to be kept. GNN-FiLM中提出的
        self_edge_vec[:, self.n_etype] = 1

        head_type = node_type[edge_index[0]]  # [E,] #head=src
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat(
            [self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec],
                             dim=0)  # [E+N, ?] ?等于39吧
        headtail_vec = torch.cat(
            [headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?] ?等于8
        # print("edge_vec:", edge_vec.size())         edge_vec: torch.Size([9964, 39])
        # print("headtail_vec:", headtail_vec.size()) headtail_vec: torch.Size([9964, 8])

        node_batch = torch.tensor([[i]*node_feature_extra[3] for i in range(
            node_feature_extra[2])]).view(1, -1).to(node_type.device)
        node_feature_extra = torch.cat([node_feature_extra[0], node_feature_extra[1]], dim=2).view(
            node_type.size(0), -1).contiguous()  # [`total_n_nodes`, dim] [2000,200]

        # [E+N, emb_dim] 这里的edge的编码是融合了边的type和头尾节点type
        edge_embeddings = self.edge_encoder(
            torch.cat([edge_vec, headtail_vec], dim=1))

        '''
        # 进行边的加权，求权重系数  size mismatch, m1: [27160 x 400], m2: [200 x 200] at
            (Pdb) torch.Size([16784, 1])
            (Pdb) torch.Size([14784])
            这里可以解释为sent_vecs的指导剪枝操作，指导边的剪枝；x为节点的初始化值来确定不同的节点类型的重要性
        '''
        node_batch = ((edge_batch[-1]+torch.tensor(1)) +
                      node_batch).to(node_type.device)

        # edge_embeddings = edge_embeddings * normalized_wts + edge_embeddings
        # pooled_edge_vecs = scatter_sum(edge_embeddings * normalized_wts, edge_node_batch, dim=0, dim_size=int(N/200))

        # Add self loops to edge_index
        loop_index = torch.arange(
            0, N, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)  # [2, N]
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        # 这里来用sent_vec来加强节点的嵌入表示 （效果不佳）
        '''
        sent_hidden = self.sent2hidden(sent_vecs).unsqueeze(1) # [bs, 1, emb_dim]
        x = x.view(sent_hidden.size(0), -1, sent_hidden.size(-1)).contiguous()
        nodeAtt = torch.sum(sent_hidden*x, dim=-1, keepdim=True) # [ba, node, emb_dim]
        nodeAtt = torch.softmax(nodeAtt, dim=-2) #[bs, N, 1]
        x = x*nodeAtt
        x = x.view(-1, x.size(2)).contiguous() # view(-1, x.size(-1)).constiguous() 
        '''

        # print("sent_hidden_size:", sent_hidden.size())
        # print("x size:", x.size())
        # print("node_feature_extra size:", node_feature_extra.size())
        # sent_hidden_size: torch.Size([10, 1, 200])
        # x size: torch.Size([10, 2000, 200])
        # node_feature_extra size: torch.Size([2000, 200])

        self_x = x
        # [N, 2*emb_dim] -> [2000,400]
        x = torch.cat([x, node_feature_extra], dim=1)

        x = (x, x)
        # 加了自环的边索引，结点的编码以及边的编码(来源于边类型与边两端头尾结点的类型)
        # 在内部调用message()、aggregate()和update()函数。作为消息传播的额外参数，传入节点嵌入x和规范化系数norm
        # 传入了 edge_index (包含了自环)

        extra = (sent_vecs, edge_batch, node_batch)

        # [2, E+N], [N, 2*emb_dim], [E+N, emb_dim] -> [N, emb_dim]
        aggr_out = self.propagate(
            edge_index, x=x, edge_attr=edge_embeddings, extra=extra)

        # aggr_out [2000,200]
        '''
        Gated 0311-2054这次是在self-att基础上再进行 gate操作模型, 目前是暂时不加上这个门机制效果更好
        
        m = torch.sigmoid(self.linear1(aggr_out))
        aggr_out = m*self_x + (1-m)*self.linear2(aggr_out)
        '''
        out = self.mlp(aggr_out)

        # 重新转换成三维的形状
        '''
        out = out.view(, -1) # node_type.size(0), node_type.size(1)
        node_type = node_type.view()
        node_feature_extra = node_feature_extra.view()
        '''

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out  # , edge_index, edge_type, node_type, node_feature_extra

    # i: tgt, j:src   Node j -> Node i x_j 是row | x_i 就相等于邻居节点
    def message(self, edge_index, x_i, x_j, edge_attr, extra):
        # edge_attr.size() torch.Size([10166, 200]) E+N = 10166
        # edge_index.size() torch.Size([2, 10166])
        # x_j.size() torch.Size([10166, 400])
        # x_i.size() torch.Size([10166, 400])
        # wts.size() [E+N, 1]

        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim  # edge_embedding
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(
            0) == edge_index.size(1)  # N

        '''
        这里可以先进行相关权重计算 首先是边的权重信息
        改进了key不加edge_attr
        我认为还应该加上有关于问题中的中心实体和句子整体的关系和 这里应该可以得到三个矩阵 [N,5] N是句子中心实体的个数-> torch.sum(dim=0)[1,5]
        '''

        sent_vecs = extra[0]  # [10, 200]
        edge_batch = extra[1]  # [E, ]
        node_batch = extra[2]  # [1, N(2000)]
        # out1 = torch.cat([edge_attr, torch.cat([sent_vecs[edge_batch], x_j], dim=0)], dim=1)
        # 试试吧从x_j改成x_i x_i目前来看效果还是差一些
        out1 = torch.cat([edge_attr, x_j], dim=1)
        wts = self.weight_mlp(out1)
        unnormalized_wts = wts

        edge_node_batch = torch.cat([edge_batch.unsqueeze(
            0), node_batch], dim=1).squeeze_(0)  # [E+N, ]
        wts = scatter_softmax(wts.squeeze(1), edge_node_batch, dim=0)
        normalized_wts = wts.unsqueeze(1)  # [E+N, 1]
        edge_attr = edge_attr*normalized_wts  # [E+N, 200] * [E+N, 1]

        # [N, emb_dim],[N, 2*emb_dim] -> [E, heads, _dim]
        key = self.linear_key2(
            x_i).view(-1, self.head_count, self.dim_per_head)
        # key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) # [N, emb_dim],[N, 2*emb_dim] -> [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(
            -1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(
            x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        # [E, heads] #group by src side node
        alpha = softmax(scores, src_node_index)
        self._alpha = alpha

        '''
        这里暂时不考虑src节点得出度特征,但是还是有这个dgree的特征效果好
        '''

        # adjust by outgoing degree of src 这里计算的是出度
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(
            ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(
            src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]

        # [bs*node_num, emb_dim]
        return out.view(-1, self.head_count * self.dim_per_head)
