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
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        streamer=None,
        assistant_model: Optional["PreTrainedModel"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        if inputs_embeds is None:
            inputs_embeds = self.custom_linear(sign_inputs)
        
        return self.model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            streamer=streamer,
            assistant_model=assistant_model,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )
    
    # Override the forward method to modify input embeddings
    def forward(
        self,
        sign_inputs=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # cache_position=None,
    ):
        
        # Apply custom linear layer to the input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.custom_linear(sign_inputs)

        # Pass modified embeddings to the original T5 forward method
        return self.model.forward(
            input_ids=None,  # We use inputs_embeds instead of input_ids
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position=cache_position,
        )
