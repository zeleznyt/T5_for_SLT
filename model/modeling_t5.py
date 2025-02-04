import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    AutoModelForSeq2SeqLM
)
from .configuration_t5 import SignT5Config
from transformers.generation.utils import GenerationConfig, LogitsProcessorList, StoppingCriteriaList, GenerateOutput
from typing import Optional, Union, List, Callable

# Subclassing the T5 model
class T5ModelForSLT(PreTrainedModel):

    config_class = SignT5Config

    def __init__(self, config: SignT5Config):
        super().__init__(config)

        # Define a custom linear layer to apply to the input embeddings
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name)
        self.custom_linear = nn.Sequential(
            nn.Linear(config.sign_input_dim, self.model.config.d_model),
            nn.Dropout(config.hidden_dropout_prob),
            nn.GELU(),
        )

        self.model.generation_config = GenerationConfig(
            max_length=config.max_length,
            num_beams=config.num_beams,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            do_sample=config.do_sample,
            early_stopping=config.early_stopping,
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            decoder_start_token_id=self.model.config.pad_token_id,
        )

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