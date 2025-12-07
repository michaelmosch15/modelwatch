from __future__ import annotations
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class GPT2WithPIIFiltering(GPT2LMHeadModel):

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        
        self.pii_filter_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.n_embd, 
                nhead=config.n_head, 
                dim_feedforward=config.n_inner if config.n_inner else 4 * config.n_embd,
                dropout=config.resid_pdrop,
                activation="gelu",
                layer_norm_eps=config.layer_norm_epsilon,
                batch_first=True
            ),
            num_layers=6
        )
        
        self.pii_classifier = nn.Linear(config.n_embd, 1)
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        
        self._init_weights(self.pii_filter_layer)
        self._init_weights(self.pii_classifier)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)

        filter_features = self.pii_filter_layer(inputs_embeds)
        
        pii_logits = self.pii_classifier(filter_features)
        pii_probs = torch.sigmoid(pii_logits)

        filtered_embeds = (1 - pii_probs) * inputs_embeds + (pii_probs) * self.mask_embedding

        return super().forward(
            input_ids=None, 
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=filtered_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
