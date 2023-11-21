"""
This module provides the functions and classes for running Monitor-Guided Decoding over OpenAI models
"""

from enum import Enum
import time
from typing import List, Set
import torch
import asyncio

from openai import OpenAI
from monitors4codegen.monitor_guided_decoding.monitor import Monitor
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import TikTokenWrapper

class OpenAI_Models(Enum):
    TD3 = 'text-davinci-003'

def openai_mgd(
    client: OpenAI,
    model: OpenAI_Models,
    tokenizer: TikTokenWrapper,
    prompt_tokenized: torch.Tensor,
    temp: float,
    top_p: float,
    monitor: Monitor,
    num_new_tokens: int
):
    """
    This function generates completions with OpenAI models using the Monitor-Guided Decoding scheme.
    """
    prompt_tokenized: torch.Tensor = torch.tensor(prompt_tokenized, dtype=torch.int64)
    assert len(prompt_tokenized.shape) == 1

    all_tokens: torch.Tensor = prompt_tokenized
    gen_text: bytes = b''
    
    gen_tokens: List[int] = []

    tokens_sort_key = {k:[0, 0] for k in tokenizer.all_token_ids}
    
    # # TODO: Find a way to prioritize tokens to be blacklisted
    
    # # Why prioritize? OpenAI allows applying logit_bias to upto 300 tokens, whereas the typical number of tokens in vocabulary is 50,000.
    # # Because of this, it is necessary to identify the top 300 tokens, that we think need to be either blacklisted, or whitelisted.
    # # This prioritization should be done taking into account what violating token is the model likely to predict in the next step.

    # # Options for prioritization of tokens:
    # # 1. The following code uses info about whether the token has a break char in it
    # for token, token_id in tokenizer.vocab_trie.iteritems():
    #     if token[0] in monitor.all_break_chars:
    #         tokens_sort_key[token_id][0] = 0 # ".", ", a"
    #     elif any([c in monitor.all_break_chars for c in token]):
    #         tokens_sort_key[token_id][0] = 1 # "abc, "
    #     else:
    #         tokens_sort_key[token_id][0] = 2

    # # 2. The following code uses frequency of the token in repo as a heuristic
    # for freq_token, freq in metadata_batch[seq_idx]['token_freq']:
    #     tokens_sort_key[freq_token][1] = freq

    # # 3. Use a local-small and very fast language model to score the tokens

    # # 4. Use the prompt to score the tokens

    all_text_bytes: bytes = tokenizer.tokenizer.decode_bytes(all_tokens.tolist())
    prompt_num_tokens: int = all_tokens.shape[0]

    priority_blacklist: List[int] = []

    while all_tokens.shape[0] < prompt_num_tokens + num_new_tokens:
        num_toks_to_gen = (prompt_num_tokens + num_new_tokens) - all_tokens.shape[0]
        
        blacklisted_ids: List[int] = asyncio.run_coroutine_threadsafe(monitor.maskgen(all_tokens.tolist()), monitor.monitor_file_buffer.lsp.server.loop).result()
        white_listed_ids: Set[int] = set(tokenizer.all_token_ids) - set(blacklisted_ids+[50256])

        logit_bias = {50256:-100}

        for token_id in priority_blacklist:
            logit_bias[token_id] = -100

        if len(white_listed_ids) <= (300 - len(logit_bias)):
            for white_token_id in white_listed_ids:
                logit_bias[white_token_id] = 100
        else:
            for candidate_token in sorted(blacklisted_ids, key=lambda x: tokens_sort_key[x], reverse=True):
                if len(logit_bias) >= 300:
                    break
                if candidate_token in blacklisted_ids:
                    logit_bias[candidate_token] = -100

        exponential_backoff_wait = 1
        while True:
            try:
                prompt_arg: str = all_text_bytes.decode('utf-8', errors='strict')
            except UnicodeDecodeError:
                prompt_arg: List[int] = all_tokens.tolist()
            
            try:
                response = client.completions.create(
                    model=model.value,
                    prompt=[prompt_arg],
                    temperature=temp,
                    max_tokens=num_toks_to_gen if len(logit_bias) <= 1 else 1,
                    top_p=top_p,
                    stop=['.'],
                    logit_bias=logit_bias,
                    logprobs=5
                )
                break
            except Exception:
                time.sleep(exponential_backoff_wait)
                if exponential_backoff_wait < 64:
                    exponential_backoff_wait = exponential_backoff_wait*2
                else:
                    exponential_backoff_wait = 1

        assert len(response.choices) == 1

        def convert_bytesrep_to_bytes(x: str) -> bytes:
            if x.startswith('bytes:'):
                return bytes.fromhex(x.replace('bytes:', '').replace('\\x', ''))
            else:
                return x.encode()

        tokens_gen_bytes_ = list(map(convert_bytesrep_to_bytes, response.choices[0].logprobs.tokens))
        tokens_gen_bytes = []
        dot_found = False
        for token_bytes in tokens_gen_bytes_:
            gen_text += token_bytes
            all_text_bytes += token_bytes
            tokens_gen_bytes.append(token_bytes)
            if b'.' in token_bytes:
                dot_found = True
                break

        # When "stop" sequence is sent to openai model, it will not generate text beyond the text sequence within the "stop" parameter.
        # However, when it stops because of the "stop" sequence, the returned text does not contain the stop sequence, and only includes
        # text upto the stop sequence. So, the following code determines if the stop sequence "." needs to be added manually.
        should_manually_add_dot = None
        if response.choices[0].finish_reason == 'stop':
            if dot_found:
                should_manually_add_dot = False
            else:
                should_manually_add_dot = True
        elif response.choices[0].finish_reason == 'length':
            should_manually_add_dot = False
        else:
            raise Exception("Unknown finish reason", response.choices[0].finish_reason)

        tokens_gen = list(map(lambda x: tokenizer.tokenizer.encode_single_token(x), tokens_gen_bytes))

        assert should_manually_add_dot is not None
        if should_manually_add_dot:
            gen_text += b'.'
            all_text_bytes += b'.'
            tokens_gen.append(tokenizer.tokenizer.encode_single_token('.'))
        
        if len(logit_bias) > 1:
            assert len(tokens_gen) == 1, (print(response), response, launch_debug(locals()))
            if tokens_gen[0] in blacklisted_ids:
                priority_blacklist.append(tokens_gen[0])
                continue
        priority_blacklist = []

        new_all_tokens = torch.cat([
            all_tokens,
            torch.tensor(tokens_gen)
        ]).to(all_tokens)

        assert len(new_all_tokens.shape) == 1
        assert new_all_tokens.shape[0] > all_tokens.shape[0], (new_all_tokens.shape, all_tokens.shape, launch_debug(locals()))
        assert torch.equal(new_all_tokens[:all_tokens.shape[0]], all_tokens)
        gen_tokens += new_all_tokens[all_tokens.shape[0]:].tolist()
        all_tokens = new_all_tokens

    return gen_tokens, gen_text.decode()
