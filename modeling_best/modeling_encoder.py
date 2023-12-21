import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
try:
    from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
except:
    pass
from transformers import AutoModel, BertModel, BertConfig, AutoConfig
# from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from utils_best.layers import *
from utils_best.data_utils import get_gpt_token_num

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()) + ['LIAMF-USP/aristo-roberta'],
    'lstm': ['lstm'],
}
try:
    MODEL_CLASS_TO_NAME['albert'] = list(
        ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
except:
    pass

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class,
                       model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}

# Add SapBERT configuration, 实际上并不是在这个预训练后的plm上进行实验,是额外增加了一些训练数据
model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_TO_CLASS[model_name] = 'bert'


class LSTMTextEncoder(nn.Module):
    pool_layer_classes = {'mean': MeanPoolLayer, 'max': MaxPoolLayer}

    def __init__(self, vocab_size=1, emb_size=300, hidden_size=300, output_size=300, num_layers=2, bidirectional=True,
                 emb_p=0.0, input_p=0.0, hidden_p=0.0, pretrained_emb_or_path=None, freeze_emb=True,
                 pool_function='max', output_hidden_states=False):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.output_hidden_states = output_hidden_states
        assert not bidirectional or hidden_size % 2 == 0

        if pretrained_emb_or_path is not None:
            # load pretrained embedding from a .npy file
            if isinstance(pretrained_emb_or_path, str):
                pretrained_emb_or_path = torch.tensor(
                    np.load(pretrained_emb_or_path), dtype=torch.float)
            emb = nn.Embedding.from_pretrained(
                pretrained_emb_or_path, freeze=freeze_emb)
            emb_size = emb.weight.size(1)
        else:
            emb = nn.Embedding(vocab_size, emb_size)
        self.emb = EmbeddingDropout(emb, emb_p)
        self.rnns = nn.ModuleList([nn.LSTM(emb_size if l == 0 else hidden_size,
                                           (hidden_size if l != num_layers else output_size) // (
                                               2 if bidirectional else 1),
                                           1, bidirectional=bidirectional, batch_first=True) for l in range(num_layers)])
        self.pooler = self.pool_layer_classes[pool_function]()

        self.input_dropout = nn.Dropout(input_p)
        self.hidden_dropout = nn.ModuleList(
            [RNNDropout(hidden_p) for _ in range(num_layers)])

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        assert (lengths > 0).all()
        batch_size, seq_len = inputs.size()
        hidden_states = self.input_dropout(self.emb(inputs))
        all_hidden_states = [hidden_states]
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dropout)):
            hidden_states = pack_padded_sequence(
                hidden_states, lengths, batch_first=True, enforce_sorted=False)
            hidden_states, _ = rnn(hidden_states)
            hidden_states, _ = pad_packed_sequence(
                hidden_states, batch_first=True, total_length=seq_len)
            all_hidden_states.append(hidden_states)
            if l != self.num_layers - 1:
                hidden_states = hid_dp(hidden_states)
        pooled = self.pooler(all_hidden_states[-1], lengths)
        assert len(all_hidden_states) == self.num_layers + 1
        outputs = (all_hidden_states[-1], pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs


class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states
        assert not self.output_token_states or self.model_type in (
            'bert', 'roberta', 'albert')

        if self.model_type in ('lstm',):
            self.module = LSTMTextEncoder(**kwargs, output_hidden_states=True)
            self.sent_dim = self.module.output_size
        else:
            model_class = AutoModel
            print('model_name:', model_name)
            # 这里设置了output_hidden_states, 因此最后一层就是(last_hidden_state, pooler_output, hidden_states)
            # last_hidden_state (batch_size, sequence_length, hidden_size) pooler_output(batch_size, hidden_size) hidden_states (batch_size, sequence_length, hidden_size)
            model_name = '/data1/kbqa/xmh/Pretrained/roberta-large'
            config = AutoConfig.from_pretrained(
                model_name, output_hidden_states=True)
            self.module = model_class.from_pretrained(
                model_name, config=config)

            # self.module = model_class.from_pretrained(model_name, output_hidden_states=True)
            if from_checkpoint is not None:
                self.module = self.module.from_pretrained(
                    from_checkpoint, output_hidden_states=True)
            if self.model_type in ('gpt',):
                self.module.resize_token_embeddings(get_gpt_token_num())
            # roberta-large hidden_size = 1024
            self.sent_dim = self.module.config.n_embd if self.model_type in (
                'gpt',) else self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        '''
        layer_id: only works for non-LSTM encoders
        output_token_states: if True, return hidden states of specific layer and attention masks
        '''

        if self.model_type in ('lstm',):  # lstm
            input_ids, lengths = inputs
            outputs = self.module(input_ids, lengths)
        elif self.model_type in ('gpt',):  # gpt
            input_ids, cls_token_ids, lm_labels = inputs  # lm_labels is not used
            outputs = self.module(input_ids)
        else:  # bert / xlnet / roberta
            # input_ids, input_mask(1位置表示有值), segment_ids(区分前后的token), output_mask(有值的地方为0)
            input_ids, attention_mask, token_type_ids, output_mask = inputs
            # mini-batch*num_choices=10
            # print('input_ids:', input_ids.size()) torch.Size([10, 100])
            outputs = self.module(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        all_hidden_states = outputs[-1]
        # final layer output [bs, seq_len, hidden_dim]
        hidden_states = all_hidden_states[layer_id]
        # print("output_size:", len(outputs)) # output_size=3 [last_hidden_state, pooler_output, hidden_states]
        # print("all_hidden_states:", all_hidden_states.size())
        # print('hidden_states_size:', hidden_states.size())
        if self.model_type in ('lstm',):
            sent_vecs = outputs[1]
        elif self.model_type in ('gpt',):
            cls_token_ids = cls_token_ids.view(
                -1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            sent_vecs = hidden_states.gather(1, cls_token_ids).squeeze(1)
        elif self.model_type in ('xlnet',):
            sent_vecs = hidden_states[:, -1]
        elif self.model_type in ('albert',):
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = hidden_states[:, 0]
        else:  # bert / roberta
            if self.output_token_states:
                return hidden_states, output_mask
                # 对最后一层的输出进行池化 (bsz, seq_len, hidden_size)
            sent_vecs = self.module.pooler(hidden_states)

        # print('sent_vecs:', sent_vecs.size()) # [10, 1024] = [mini_bs*nc, sent_dim]
        return sent_vecs, all_hidden_states, attention_mask


def run_test():
    encoder = TextEncoder('lstm', vocab_size=100,
                          emb_size=100, hidden_size=200, num_layers=4)
    input_ids = torch.randint(0, 100, (30, 70))
    lenghts = torch.randint(1, 70, (30,))
    outputs = encoder(input_ids, lenghts)
    assert outputs[0].size() == (30, 200)
    assert len(outputs[1]) == 4 + 1
    assert all([x.size() == (30, 70, 100 if l == 0 else 200)
               for l, x in enumerate(outputs[1])])
    print('all tests are passed')
