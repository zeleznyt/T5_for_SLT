from transformers import PretrainedConfig
from typing import List


class SignT5Config(PretrainedConfig):

    model_type = "t5"

    def __init__(
            self,
            base_model_name: str = "t5-small",
            sign_input_dim=208,
            hidden_dropout_prob=0.0,
            max_length=128,  # from YT-ASL paper "decoder context window"
            num_beams=1,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
            do_sample=False,
            early_stopping=False,
            **kwargs

            # freeze_shared=False,
            # pad_token_id: int = 0,
            # bos_token_id: int = 1,
            # eos_token_id: int = 2,
            # generation parameters
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

        # Unused:
        # self.freeze_shared = freeze_shared
        #
        # self.pad_token_id = pad_token_id
        # self.bos_token_id = bos_token_id
        # self.eos_token_id = eos_token_id


        # self.max_length =
        # self.num_beams =
        # self.temperature =
        # self.top_k =
        # self.top_p =
        # self.repetition_penalty =
        # self.length_penalty =
        # self.no_repeat_ngram_size =
        # self.do_sample =
        # self.early_stopping =
        # self.base_model_name =
        # self.hidden_dropout_prob =
        # self.sign_input_dim =


        # pad_token_id = self.model.config.pad_token_id,
        # bos_token_id = self.model.config.bos_token_id,
        # eos_token_id = self.model.config.eos_token_id,
        # decoder_start_token_id = self.model.config.pad_token_id,
        # self.model.config.d_model
