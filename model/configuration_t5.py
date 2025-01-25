from transformers import PretrainedConfig
from typing import List


class SignT5Config(PretrainedConfig):

    model_type = "t5"

    def __init__(
            self,
            base_model_name: str = "t5-small",
            sign_input_dim=255,
            freeze_shared=False,
            hidden_dropout_prob=0.1,
            pad_token_id: int = 0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            # generation parameters
            num_beams=4,
            max_length=128,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            length_penalty=2.0,
            repetition_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=True,
            **kwargs
    ) -> None:
        """
        Initializes SignT5 configuration.
        
        Args:
            base_model_name (str): The base model name of the T5 model.

        Returns:
            None
        """
        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.sign_input_dim = sign_input_dim
        self.hidden_dropout_prob = hidden_dropout_prob

        self.freeze_shared = freeze_shared

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.num_beams = num_beams
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.length_penalty = length_penalty    
        self.repetition_penalty = repetition_penalty
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.do_sample = do_sample