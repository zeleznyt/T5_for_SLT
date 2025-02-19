import os.path
import torch
from torch import nn
from .configuration_t5 import SignT5Config
from typing import Optional, Union, List, Callable

from transformers import T5ForConditionalGeneration, GenerationConfig, PreTrainedModel
import json

class T5ModelForSLT(PreTrainedModel):
    config_class = SignT5Config
    def __init__(self, config: SignT5Config, generation_config: Optional[GenerationConfig] = None):
        super().__init__(config)

        if config.model_path:
            model_name_or_path = config.model_path
        else:
            model_name_or_path = config.base_model_name
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=config)

        # Custom linear layer to transform sign language input embeddings
        self.custom_linear = nn.Sequential(
            nn.Linear(config.sign_input_dim, self.model.config.d_model),
            nn.Dropout(config.hidden_dropout_prob),
            nn.GELU(),
        )

        # Apply custom generation config if provided
        if generation_config:
            self.model.generation_config = GenerationConfig(**generation_config) # TODO: check if works
        elif config.model_path:
            self.model.generation_config = GenerationConfig.from_pretrained(config.model_path)

        if not self.model.config.decoder_start_token_id:
            self.model.config.decoder_start_token_id = config.pad_token_id


    @torch.no_grad()
    def generate(
            self,
            sign_inputs: torch.FloatTensor = None,
            attention_mask: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            **kwargs):
        """Overrides generate() to handle sign language inputs."""

        # If no input embeddings are provided, generate them from sign_inputs
        if inputs_embeds is None:
            inputs_embeds = self.custom_linear(sign_inputs)

        return self.model.generate(
            input_ids=None,  # We use inputs_embeds instead of input_ids
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs)

    def forward(
            self,
            sign_inputs=None,
            attention_mask=None,
            labels=None,
            inputs_embeds=None,
            **kwargs):
        """Overrides forward() to handle sign language inputs."""

        # Apply custom linear layer to the input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.custom_linear(sign_inputs)

        return self.model.forward(
            input_ids=None,  # We use inputs_embeds instead of input_ids
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            **kwargs)