"""
This file provides the tokenizer wrapper that is used to provide a common interface over 
HF tokenizers and TikToken tokenizers
"""

import torch
import tiktoken

from typing import List, Set, Union
from pygtrie import CharTrie
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TokenizerWrapper:
    """
    This class provides a common interface over HF tokenizers and TikToken tokenizers
    """

    def __init__(self, tokenizer: Union[PreTrainedTokenizerBase, tiktoken.core.Encoding]):
        """
        Initializes the tokenizer wrapper
        """
        self.tokenizer = tokenizer
        self.vocab_trie = CharTrie()
        self.tokenizer_char_set: Set[str] = set()
        self.all_token_ids: Set[int] = set()

    def decode(self, *args, **kwargs) -> str:
        """
        Decodes the given token ids to a string

        Params:
        token_ids, clean_up_tokenization_spaces, skip_special_tokens
        """
        raise NotImplementedError()

    def convert_ids_to_tokens(self, x) -> List[str]:
        """
        Converts the given token ids to a list of tokens
        """
        raise NotImplementedError()

    def convert_tokens_to_string(self, x) -> str:
        """
        Converts the given list of tokens to a string
        """
        raise NotImplementedError()


class HFTokenizerWrapper(TokenizerWrapper):
    """
    This class provides an instance of TokenizerWrapper for HF tokenizers
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.__dict__.update(tokenizer.__dict__)
        for k, v in tokenizer.vocab.items():
            decoded_token = tokenizer.decode(v, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            if decoded_token != "":
                self.tokenizer_char_set.update(decoded_token)
                self.vocab_trie[decoded_token] = v
        self.all_token_ids = set(tokenizer.vocab.values())

    def decode(self, *args, **kwargs) -> str:
        """
        Decodes the given token ids to a string
        """
        return self.tokenizer.decode(*args, **kwargs)

    def convert_ids_to_tokens(self, x) -> List[str]:
        """
        Converts the given token ids to a list of tokens
        """
        return self.tokenizer.convert_ids_to_tokens(x)

    def convert_tokens_to_string(self, x) -> str:
        """
        Converts the given list of tokens to a string
        """
        return self.tokenizer.convert_tokens_to_string(x)


class TikTokenWrapper(TokenizerWrapper):
    """
    This class provides an instance of TokenizerWrapper for TikToken tokenizers
    """
    def __init__(self, tokenizer: tiktoken.core.Encoding):
        super().__init__(tokenizer)

        assert len(tokenizer.special_tokens_set) == 1
        self.all_special_ids = {tokenizer.encode_single_token(token) for token in tokenizer.special_tokens_set}
        for k_ in tokenizer.token_byte_values():
            v = tokenizer.encode_single_token(k_)
            decoded_token = tokenizer.decode([tokenizer.encode_single_token(k_)])
            if decoded_token != "":
                self.tokenizer_char_set.update(decoded_token)
                self.vocab_trie[decoded_token] = v
                self.all_token_ids.add(v)

    def decode(self, token_ids: Union[List[int], torch.Tensor], *args, **kwargs) -> str:
        """
        Decodes the given token ids to a string
        """
        clean_up_tokenization_spaces, skip_special_tokens = None, None
        if len(args) == 0:
            pass
        elif len(args) == 1:
            skip_special_tokens: bool = args[0]
        elif len(args) == 2:
            skip_special_tokens, clean_up_tokenization_spaces = args[0], args[1]

        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = kwargs.get("clean_up_tokenization_spaces", True)
        if skip_special_tokens is None:
            skip_special_tokens = kwargs.get("skip_special_tokens", False)

        assert not clean_up_tokenization_spaces
        assert skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        token_ids: List[int] = [i for i in token_ids if i not in self.all_special_ids]

        return self.tokenizer.decode(token_ids)

    def convert_ids_to_tokens(self, x) -> List[str]:
        """
        Converts the given token ids to a list of tokens
        """
        return [self.tokenizer.decode([i]) for i in x]

    def convert_tokens_to_string(self, x) -> str:
        """
        Converts the given list of tokens to a string
        """
        return "".join(x)
