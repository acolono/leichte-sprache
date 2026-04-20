"""PyTorch pronoun classifier model.

1:1 translation of the MLX model from the personalpronomen package.
Use model.eval() before inference to disable dropout.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the pronoun classifier model.

    Class count defaults match the ACTUAL trained model (from label_encoders.json),
    NOT the original MLX ModelConfig defaults which had incorrect values for
    num_case_classes (5 instead of 4) and num_polite_classes (3 instead of 2).
    """

    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    vocab_size: int = 30000
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12

    # Pronoun classification heads (correct counts from label_encoders.json)
    num_person_classes: int = 4
    num_gender_classes: int = 4
    num_number_classes: int = 3
    num_case_classes: int = 4      # NOT 5 -- actual trained model uses 4
    num_polite_classes: int = 2    # NOT 3 -- actual trained model uses 2
    num_reflex_classes: int = 2


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer (custom implementation matching MLX mask convention)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        # Linear transformations
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Reshape for multi-head attention: [batch, seq, heads, head_size] -> [batch, heads, seq, head_size]
        query_layer = query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Attention scores: [batch, heads, seq, seq]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask (same convention as MLX: 1=attend, 0=mask)
        if attention_mask is not None:
            attention_scores = attention_scores + (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # Softmax + dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)

        return context_layer


class TransformerLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_norm(attention_output + hidden_states)

        # Feed-forward with GELU (exact, not approximate, to match MLX nn.gelu)
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.gelu(intermediate_output, approximate="none")
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_norm(layer_output + attention_output)

        return layer_output


class PronounClassifier(nn.Module):
    """Multi-task pronoun classifier model.

    PyTorch equivalent of the MLX PronounClassifier.
    Call model.eval() before inference to disable dropout.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer layers -- use setattr to match NPZ key names (encoder_layer_0, etc.)
        # Do NOT create self.encoder_layers list (would produce duplicate state_dict keys)
        self.num_layers = config.num_hidden_layers
        for i in range(config.num_hidden_layers):
            setattr(self, f"encoder_layer_{i}", TransformerLayer(config))

        # Pronoun-specific pooling
        self.pronoun_pooler = nn.Linear(config.hidden_size, config.hidden_size)

        # Classification heads
        self.person_classifier = nn.Linear(config.hidden_size, config.num_person_classes)
        self.gender_classifier = nn.Linear(config.hidden_size, config.num_gender_classes)
        self.number_classifier = nn.Linear(config.hidden_size, config.num_number_classes)
        self.case_classifier = nn.Linear(config.hidden_size, config.num_case_classes)
        self.polite_classifier = nn.Linear(config.hidden_size, config.num_polite_classes)
        self.reflex_classifier = nn.Linear(config.hidden_size, config.num_reflex_classes)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings (word + position + token_type)."""
        seq_length = input_ids.shape[1]

        word_embeds = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pronoun_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            pronoun_mask: Mask indicating pronoun positions [batch_size, seq_length]
            candidate_mask: Mask indicating candidate antecedent positions [batch_size, seq_length]

        Returns:
            Dictionary with logits for each classification task
        """
        # Get embeddings
        hidden_states = self.get_embeddings(input_ids)

        # Pass through transformer layers
        for i in range(self.num_layers):
            hidden_states = getattr(self, f"encoder_layer_{i}")(hidden_states, attention_mask)

        # Pool pronoun representations
        if pronoun_mask is not None:
            pronoun_mask_expanded = pronoun_mask[:, :, None]
            masked_hidden = hidden_states * pronoun_mask_expanded
            pronoun_lengths = pronoun_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pronoun_repr = masked_hidden.sum(dim=1) / pronoun_lengths
        else:
            # Use CLS token representation as fallback
            pronoun_repr = hidden_states[:, 0, :]

        # Enhance pronoun representation with candidate antecedent context
        if candidate_mask is not None and candidate_mask.sum().item() > 0:
            candidate_mask_expanded = candidate_mask[:, :, None]
            candidate_hidden = hidden_states * candidate_mask_expanded

            pronoun_repr_expanded = pronoun_repr[:, None, :]
            attention_scores = (candidate_hidden * pronoun_repr_expanded).sum(dim=-1)
            attention_scores = attention_scores * candidate_mask + (1.0 - candidate_mask) * -10000.0
            attention_weights = F.softmax(attention_scores, dim=-1)

            attention_weights_expanded = attention_weights[:, :, None]
            candidate_context = (hidden_states * attention_weights_expanded).sum(dim=1)

            pronoun_repr = pronoun_repr + 0.3 * candidate_context

        # Apply pooler
        pronoun_repr = self.pronoun_pooler(pronoun_repr)
        pronoun_repr = torch.tanh(pronoun_repr)

        # Classification heads
        logits = {
            "person": self.person_classifier(pronoun_repr),
            "gender": self.gender_classifier(pronoun_repr),
            "number": self.number_classifier(pronoun_repr),
            "case": self.case_classifier(pronoun_repr),
            "polite": self.polite_classifier(pronoun_repr),
            "reflex": self.reflex_classifier(pronoun_repr),
        }

        return logits


def create_model(config: Optional[ModelConfig] = None) -> PronounClassifier:
    """Create a pronoun classifier model.

    Args:
        config: Model configuration (uses default if None)

    Returns:
        PronounClassifier model
    """
    if config is None:
        config = ModelConfig()

    return PronounClassifier(config)
