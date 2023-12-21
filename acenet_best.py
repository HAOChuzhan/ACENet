import datetime
import subprocess
import os
import socket
import setproctitle
import numpy as np
from collections import defaultdict, OrderedDict
import random
from turtle import pd
import scipy as sp
import scipy.stats
# import numpy as np
import pdb
try:
    from transformers import (
        ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup

from modeling_best.modeling_acenet_best import *
from utils_best.optimization_utils import OPTIMIZER_CLASSES
from utils_best.parser_utils import *
from utils_best.utils import *


DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    # 'csqa': 3e-4, #适合采样系数小的情况
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

setproctitle.setproctitle("pythonQA-h3-head2k5-best")  # 设置进程的名称

print(socket.gethostname())
print("pid:", os.getpid())
print("conda env:", os.environ['CONDA_DEFAULT_ENV'])
print("screen: %s" % subprocess.check_output(
    'echo $STY', shell=True).decode('utf'))
print("gpu: %s" % subprocess.check_output(
    'echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))

print(DECODER_DEFAULT_LR)


def mean_confidence_interval(data, confidence=0.95):
    a = [1.0*np.array(data[i].cpu()) for i in range(len(data))]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    res = []
    res_choice = []
    model.eval()
    i = 0
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            i += 1
            logits, _ = model(*input_data)
            # print(f"{i}轮的结果:{logits}")
            # pdb.set_trace()
            n_correct += (logits.argmax(1) == labels).sum().item()
            res.extend(logits.argmax(1) == labels)
            res_choice.extend(logits.argmax(1))
            n_samples += labels.size(0)
        print('| dev/test_size=', len(res), end=" ")
        acc, h = mean_confidence_interval(res)
    res = [int(i.cpu().item()) for i in res]
    res_choice = [chr(ord('A') + int(i.cpu().item())) for i in res_choice]
    print(res)
    print(res_choice)
    return acc, h  # n_correct / n_samples


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train',
                        choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument(
        '--save_dir', default=f'./saved_models/acenet/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')

    parser.add_argument('--load_model_path', default='')

    # data
    parser.add_argument('--num_relation', default=38,
                        type=int, help='number of relations')
    parser.add_argument(
        '--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument(
        '--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument(
        '--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?',
                        const=True, help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int,
                        help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2,
                        type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=100, type=int,
                        help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int,
                        help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0,
                        type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag,
                        nargs='?', const=True, help='freeze entity embedding layer')

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False,
                        type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float,
                        help='different size of training data')
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')

    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2,
                        help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float,
                        default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2,
                        help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr',
                        default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=1, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=False, type=bool_flag,
                        help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch',
                        default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch',
                        default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help',
                        default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    args.fp16 = False  # args.fp16 and (torch.__version__ >= '1.6.0')

    if args.dataset == 'csqa':
        args.num_choice = 5
    elif args.dataset == 'obqa':
        args.num_choice = 4

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval_detail':
        # raise NotImplementedError
        eval_detail(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)

    # 随机数的固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step, dev_acc, dev_confd, test_acc, test_confd\n')
    print("Loading data ···")
    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} | concept_dim: {}'.format(concept_num, concept_dim))
    rel_emb = np.load(args.rel_emb)
    rel_extra = np.random.normal(
        0, 1, (args.num_relation-2*len(rel_emb), len(rel_emb[0])))
    rel_emb = np.concatenate((rel_emb, -rel_emb, rel_extra), 0)
    rel_emb = torch.tensor(rel_emb)
    relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
    print('| num_rels: {} | rel_dim: {}'.format(relation_num, relation_dim))

    # try:
    if True:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")

        # Dataset Construct
        dataset = LM_ACENet_DataLoader(args, args.train_statements, args.train_adj,
                                       args.dev_statements, args.dev_adj,
                                       args.test_statements, args.test_adj,
                                       batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                       device=(device0, device1),
                                       model_name=args.encoder,
                                       max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                       is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                       subsample=args.subsample, use_cache=args.use_cache)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################
        print('args.num_relation', args.num_relation)
        print('args.num_choice', args.num_choice)
        print('args.subsample', args.subsample)
        # 初始化了LM和GNN的模型
        model = LM_ACENet(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation, relation_dim=relation_dim, n_concept=concept_num,
                          concept_dim=args.gnn_dim,  # 200
                          concept_in_dim=concept_dim,  # 1024
                          mini_bs=args.mini_batch_size,
                          n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                          p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                          # 保证entity的embedding冻结不变
                          pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb, freeze_ent_emb=args.freeze_ent_emb,
                          init_range=args.init_range,  # 初始化的标准差为0.02
                          encoder_config={})
        if args.load_model_path:
            print(
                f'loading and initializing model from {args.load_model_path}')
            model_state_dict, old_args = torch.load(
                args.load_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_state_dict)

        model.encoder.to(device0)
        model.decoder.to(device1)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # 这里是对不同神经网络层设置不用的训练参数
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)  # radam opt

    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(
            args.n_epochs * (dataset.train_size() / args.batch_size))
        print('max_steps:', max_steps)
        try:
            scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(
                name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(
                name, param.size(), param.device))
    num_params = sum(p.numel()
                     for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    def compute_loss(logits, labels):
        if args.loss == 'margin_rank':
            num_choice = logits.size(1)
            flat_logits = logits.view(-1)
            # of length batch_size*num_choice
            correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)
            correct_logits = flat_logits[correct_mask == 1].contiguous(
            ).view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
            wrong_logits = flat_logits[correct_mask == 0]
            y = wrong_logits.new_ones((wrong_logits.size(0),))
            loss = loss_func(correct_logits, wrong_logits,
                             y)  # margin ranking loss
        elif args.loss == 'cross_entropy':
            loss = loss_func(logits, labels)
        return loss

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    if args.fp16:
        print('Using fp16 training')
        scaler = torch.cuda.amp.GradScaler()

    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)
    if True:
        # try:
        for epoch_id in range(args.n_epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            model.train()
            for qids, labels, *input_data in dataset.train():
                # input_data:
                # bsz*[batch_qids, batch_labels], [*batch_tensors0, *batch_tensors1, edge_index, edge_type]
                # pdb.set_trace()
                optimizer.zero_grad()
                bs = labels.size(0)
                # mini-batch train set mini-bts = mbs = 2
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            # 传入的是mini-batch大小的训练数据
                            logits, _ = model(
                                *[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                            loss = compute_loss(logits, labels[a:b])

                    else:
                        logits, _ = model(
                            *[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                        # print("logits_size:", logits.size())
                        # print("labels_size:",labels[a:b].size())
                        # [2 (mini_bs), 5(nc)], [2] # label [0:开始的索引]
                        loss = compute_loss(logits, labels[a:b])
                    loss = loss * (b - a) / bs
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                scheduler.step()
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = (time.time() - start_time) / \
                        args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | s/step {:7.2f} |'.format(
                        global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                    total_loss = 0
                    start_time = time.time()
                global_step += 1

            model.eval()
            dev_acc, dev_confd = evaluate_accuracy(dataset.dev(), model)
            save_test_preds = args.save_model
            # 暂时先不保存输出结果
            test_acc, test_confd = 0., 0.
            if not save_test_preds:
                test_acc, test_confd = evaluate_accuracy(
                    dataset.test(), model) if args.test_statements else 0.0
            else:
                eval_set = dataset.test()
                total_acc = []
                count = 0
                preds_path = os.path.join(
                    args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
                with open(preds_path, 'w') as f_preds:
                    with torch.no_grad():
                        for qids, labels, *input_data in tqdm(eval_set):
                            count += 1
                            logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(
                                *input_data, detail=True)
                            predictions = logits.argmax(1)  # [bsize, ]
                            # [bsize, n_choices]
                            preds_ranked = (-logits).argsort(1)
                            for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                                acc = int(pred.item() == label.item())
                                print('{},{}'.format(
                                    qid, chr(ord('A') + pred.item())), file=f_preds)
                                f_preds.flush()
                                total_acc.append(acc)
                test_acc = float(sum(total_acc))/len(total_acc)

            print('-' * 71)
            print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | dev_confd {:7.4f} | test_acc {:7.4f} | test_confd {:7.4f} |'.format(
                epoch_id, global_step, dev_acc, dev_confd, test_acc, test_confd))
            print('-' * 71)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{},{},{}\n'.format(
                    global_step, dev_acc, dev_confd, test_acc, test_confd))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save_model and epoch_id >= 10:

                    '''
                    暂时不用保存这个模型
                    '''
                    torch.save([model.state_dict(), args],
                               f"{model_path}.{epoch_id}")
                    with open(model_path + ".{}.log.txt".format(epoch_id), 'w') as f:
                        for p in model.named_parameters():
                            print(p, file=f)
                    print(f'model saved to {model_path}.{epoch_id}')
            else:
                if args.save_model and epoch_id >= 10:
                    torch.save([model.state_dict(), args],
                               f"{model_path}.{epoch_id}")
                    with open(model_path + ".{}.log.txt".format(epoch_id), 'w') as f:
                        for p in model.named_parameters():
                            print(p, file=f)
                    print(f'model saved to {model_path}.{epoch_id}')
            model.train()
            start_time = time.time()
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break
    # except (KeyboardInterrupt, RuntimeError) as e:
    #     print(e)


def eval_detail(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path

    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    model_state_dict, old_args = torch.load(
        model_path, map_location=torch.device('cpu'))
    print('old_args:', old_args)

    model = LM_ACENet(old_args, old_args.encoder, k=old_args.k, n_ntype=4, n_etype=old_args.num_relation, relation_dim=100, n_concept=concept_num,
                      concept_dim=old_args.gnn_dim,
                      concept_in_dim=concept_dim,
                      mini_bs=2,
                      n_attention_head=old_args.att_head_num, fc_dim=old_args.fc_dim, n_fc_layer=old_args.fc_layer_num,
                      p_emb=old_args.dropouti, p_gnn=old_args.dropoutg, p_fc=old_args.dropoutf,
                      pretrained_concept_emb=cp_emb, freeze_ent_emb=old_args.freeze_ent_emb,
                      init_range=old_args.init_range,
                      encoder_config={})
    model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    model.encoder.to(device0)
    model.decoder.to(device1)
    model.eval()

    statement_dic = {}
    for statement_path in (args.train_statements, args.dev_statements, args.test_statements):
        statement_dic.update(load_statement_dict(statement_path))

    # print('statement_dic=', statement_dic)
    use_contextualized = 'lm' in old_args.ent_emb
    args.inhouse = False
    print('inhouse =', args.inhouse)

    print('args.train_statements', args.train_statements)
    print('args.dev_statements', args.dev_statements)
    print('args.test_statements', args.test_statements)
    print('args.train_adj', args.train_adj)
    print('args.dev_adj', args.dev_adj)
    print('args.test_adj', args.test_adj)

    dataset = LM_ACENet_DataLoader(args, args.train_statements, args.train_adj,
                                   args.dev_statements, args.dev_adj,
                                   args.test_statements, args.test_adj,
                                   batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                   device=(device0, device1),
                                   model_name=old_args.encoder,
                                   max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                   is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                   subsample=args.subsample, use_cache=args.use_cache)

    save_test_preds = args.save_model
    # dev_acc, dev_confd = evaluate_accuracy(dataset.dev(), model) # dev_acc 0.7912
    # print('dev_acc {:7.4f} | dev_confd {:7.4f}'.format(dev_acc, dev_confd))
    test_acc, test_confd = 0., 0.
    if not save_test_preds:
        test_acc, test_confd = evaluate_accuracy(
            dataset.test(), model) if args.test_statements else 0.0
        print('test_acc {:7.4f}| test_confd {:7.4f}'.format(
            test_acc, test_confd))
    else:
        print('save the test_preds.csv')
        # args.inhouse = True時, test_acc  0.7526, 取的是inhouse數據集中的test數據集，False的時候才是 args.test_statements 數據集
        eval_set = dataset.test()
        total_acc = []
        total_acc_ans = []
        count = 0
        dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        preds_path = os.path.join(
            args.save_dir, 'test_preds_{}.csv'.format(dt))
        print(preds_path)
        with open(preds_path, 'w') as f_preds:
            with torch.no_grad():
                for qids, labels, *input_data in tqdm(eval_set):
                    count += 1
                    logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(
                        *input_data, detail=True)
                    predictions = logits.argmax(1)  # [bsize, ]
                    preds_ranked = (-logits).argsort(1)  # [bsize, n_choices]
                    for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                        acc = int(pred.item() == label.item())
                        print('{},{}'.format(
                            qid, chr(ord('A') + pred.item())), file=f_preds)
                        f_preds.flush()
                        total_acc.append(acc)
                        total_acc_ans.append(chr(ord('A') + pred.item()))
        test_acc = float(sum(total_acc))/len(total_acc)
        print(total_acc)
        print(total_acc_ans)
        print('-' * 71)
        print('test_acc {:7.4f}|test_confd {:7.4f}'.format(
            test_acc, test_confd))
        print('-' * 71)


if __name__ == '__main__':
    main()
