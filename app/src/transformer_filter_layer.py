from __future__ import annotations
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class GPT2WithPIIFiltering(GPT2LMHeadModel):
    """
    A GPT-2 model with a dedicated 'Soft Masking' PII filter.
    
    Architecture:
    1. Input Embeddings -> 
    2. PII Filter Transformer (6 Layers) -> 
    3. PII Probability Score -> 
    4. Soft Masking (Mixes Original vs. Mask Embedding) -> 
    5. Standard GPT-2
    """
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        
        # 1. The Filter Stack (The "Brain")
        # Uses attention to understand context (e.g., distinguishing "Index" numbers from "Phone" numbers)
        self.pii_filter_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.n_embd, 
                nhead=config.n_head, 
                dim_feedforward=config.n_inner if config.n_inner else 4 * config.n_embd,
                dropout=config.resid_pdrop,
                activation="gelu", # Force standard gelu for PyTorch compatibility
                layer_norm_eps=config.layer_norm_epsilon,
                batch_first=True
            ),
            num_layers=6
        )
        
        # 2. The Decision Head
        # Projects the filter's understanding into a single score: Is this PII? (0 = No, 1 = Yes)
        self.pii_classifier = nn.Linear(config.n_embd, 1)
        
        # 3. The Mask Embedding
        # A learnable vector representing "Redacted Information".
        # The model learns to substitute PII with this vector.
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        
        # Initialize weights
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
        # 1. Compute or retrieve initial embeddings (The raw data)
        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)

        # 2. Analyze with Filter Layer
        # The filter looks at the whole sequence to understand context.
        filter_features = self.pii_filter_layer(inputs_embeds)
        
        # 3. Calculate PII Probability
        # Sigmoid squashes output to 0-1 range.
        # High value (near 1) = High confidence it is PII.
        pii_logits = self.pii_classifier(filter_features)
        pii_probs = torch.sigmoid(pii_logits) # Shape: (batch, seq_len, 1)

        # 4. Soft Masking (The Filtering Step)
        # We mathematically blend the Original Embedding and the Mask Embedding.
        # If pii_prob is 1.0, the term becomes: (0 * Original) + (1 * Mask) = Mask
        # If pii_prob is 0.0, the term becomes: (1 * Original) + (0 * Mask) = Original
        filtered_embeds = (1 - pii_probs) * inputs_embeds + (pii_probs) * self.mask_embedding

        # 5. Pass filtered embeddings to the standard GPT-2 model
        # Now GPT-2 only sees the "Clean" data.
        return super().forward(
            input_ids=None, 
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=filtered_embeds, # Passing the sanitized embeddings
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
