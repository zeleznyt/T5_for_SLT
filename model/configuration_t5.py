from transformers import PretrainedConfig, T5Config
from typing import List

class SignT5Config(T5Config):

    model_type = "t5"

    def __init__(
            self,
            base_model_name: str = "t5-small",
            model_path: str = None,
            sign_input_dim=208,
            hidden_dropout_prob=0.0,
            max_length=128,  # from YT-ASL paper "decoder context window"
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
        self.model_path = model_path
        self.sign_input_dim = sign_input_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_length = max_length