# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)

import copy
from shutil import copy

import torch.nn.functional as F
from torch import nn
from transformers import T5PreTrainedModel as PreTrained
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Stack

from GatedHighway import Highway, MishFF, ReluFF


class TransformerDecoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.highway = Highway(d_model)
        self.relu_ff = ReluFF(d_model, d_model*4, 0.1)

    def forward(self, hidden_states):
        # We simply take the hidden state corresponding to the first token.
        out = hidden_states[:, 0]
        out = out + self.highway(out)
        out = self.relu_ff(out)
        return out


class TransformerForCodeClassification(PreTrained):
    def __init__(self, config):
        super().__init__(config)
        d_model= config.d_model
        vocab_size= config.vocab_size
        n_drop= config.dropout_rate 
        self.num_labels = config.num_labels

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = T5Stack(encoder_config, self.embedding)
        
        self.decoder = TransformerDecoder(d_model)
        self.dropout = nn.Dropout(n_drop)
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)

        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def forward(self, input_ids=None, attention_mask=None, labels=None):

        embedding = self.embedding(input_ids) # Transforming encoded context into embeddings
        encoder_out = self.encoder(inputs_embeds=embedding, attention_mask=attention_mask) # Transforming embeddings into Z state
        decoder_out = self.decoder(encoder_out.last_hidden_state) # Decoding Z 
        decoder_out = self.dropout(decoder_out) 
        logits = self.classifier(decoder_out)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
