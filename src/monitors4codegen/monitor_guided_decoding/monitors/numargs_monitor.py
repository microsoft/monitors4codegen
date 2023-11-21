"""
This module provides MGD for ensuring correct number of arguments in a method call
"""

import re

from dataclasses import dataclass
from typing import List, Union
from enum import Enum
from monitors4codegen.monitor_guided_decoding.mgd_utils import PLUtils
from monitors4codegen.multilspy.multilspy_utils import TextUtils
from monitors4codegen.multilspy.multilspy_types import Position
from monitors4codegen.monitor_guided_decoding.monitor import Monitor, MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import TokenizerWrapper


class DecoderStates(Enum):
    """
    Enum for the state of the decoder
    """

    UnInitialized = 0
    S0 = 1
    Constrained = 2

@dataclass
class OpenMethodState:
    """
    Stores information about an open method call
    """
    opening_pos: Position
    opening_idx: int
    num_tot_args: Union[int, None]
    num_args_left: Union[int, None]

class NumMethodArgumentsMonitor(Monitor):
    """
    Provides the monitor to ensure number of arguments in a method call is correct
    """

    def __init__(self, tokenizer: TokenizerWrapper, monitor_file_buffer: MonitorFileBuffer, responsible_for_file_buffer_state: bool = True) -> None:
        super().__init__(tokenizer, monitor_file_buffer, responsible_for_file_buffer_state)
        self.decoder_state = DecoderStates.UnInitialized
        self.prompt_len: Union[None, int] = None

        # Stack that keeps track of the number of open method calls
        # For example, if the current prompt is "abc(", then stack will have 1 entry
        # L[i] is the number of arguments left to be generated for the ith-nested method call
        self.open_method_calls: List[OpenMethodState] = []

    async def initialize(self, input_ids: List[int], input_text: str) -> None:
        """
        Initializes the monitor when it is invoked for the first time with inputs
        """
        self.prompt_len = len(input_ids)
        await self.pre()
        self.decoder_state = DecoderStates.Constrained

    async def pre(self) -> None:
        cursor_idx = TextUtils.get_index_from_line_col(
            self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path),
            self.monitor_file_buffer.current_lc[0],
            self.monitor_file_buffer.current_lc[1],
        )
        text_upto_cursor = self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path)[
            :cursor_idx
        ]
        if text_upto_cursor[-1] != "(":
            return

        num_args_for_currently_opened_func = await self.a_phi()

        lsp_text = self.monitor_file_buffer.lsp.get_open_file_text(
            self.monitor_file_buffer.file_path
        )
        pos_idx = TextUtils.get_index_from_line_col(
            lsp_text,
            self.monitor_file_buffer.current_lc[0],
            self.monitor_file_buffer.current_lc[1]
        )
        assert lsp_text[pos_idx-1] == "("

        if num_args_for_currently_opened_func is None:
            self.open_method_calls.append(
                OpenMethodState(
                    Position(line=self.monitor_file_buffer.current_lc[0], character=self.monitor_file_buffer.current_lc[1]),
                    pos_idx-1,
                    None,
                    None,
                )
            )
        else:
            self.open_method_calls.append(
                OpenMethodState(
                    Position(line=self.monitor_file_buffer.current_lc[0], character=self.monitor_file_buffer.current_lc[1]),
                    pos_idx-1,
                    num_args_for_currently_opened_func,
                    num_args_for_currently_opened_func,
                )
            )

    def violates(self, token: str, state: str) -> bool:
        """
        Checks if the token is in violation with the current state of the decoder
        """
        num_closeable = len(state.split('o')[0])
        num_token_closes = 0
        stack = []
        for c in token:
            if c == '(':
                stack.append('(')
            elif c == ')':
                if len(stack) == 0:
                    num_token_closes += 1
                else:
                    stack.pop()
        if num_token_closes > num_closeable:
            return True
        return False

    async def maskgen(self, input_ids: List[int]) -> List[int]:
        """
        Takes the list of input tokens, and returns the list of tokens to be blacklisted in the next step
        """
        # maskgen is invoked for every new token to be generated
        # The first time it is invoked, maskgen performs the initialization
        # Subsequent invocations are handled based on the current state of the decoder

        input_text = self.tokenizer.decode(
            input_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )

        if self.decoder_state == DecoderStates.UnInitialized:
            # Handle initialization. This is the first time monitor is being invoked
            await self.initialize(input_ids, input_text)
        else:
            # A new token has been generated. Handle the new token by calling update
            gen_so_far = self.tokenizer.decode(
                input_ids[self.prompt_len :], clean_up_tokenization_spaces=False, skip_special_tokens=True
            )
            assert gen_so_far.startswith(self.monitor_file_buffer.gen_text), (gen_so_far, self.monitor_file_buffer.gen_text)
            assert input_text.endswith(gen_so_far)
            new_gen_text = gen_so_far[len(self.monitor_file_buffer.gen_text) :]

            await self.update(new_gen_text)
        
        state = ""
        for open_method_call in self.open_method_calls[::-1]:
            if open_method_call.num_args_left is None:
                state += "c"
            else:
                if open_method_call.num_args_left == 0:
                    state += "c"
                else:
                    state += "o"
        
        blacklisted_ids = []
        for token, token_id in self.tokenizer.vocab_trie.iteritems():
            token: str = token
            if self.violates(token, state):
                blacklisted_ids.append(token_id)
        return blacklisted_ids

    async def a_phi(self) -> Union[int, None]:
        """
        Determine the number of arguments that the function which is opened at the current cursor position takes
        Assume that there can only be fix number of arguments (not variable like in case of overloading)
        """
        # This function gets called only when there's a '(' at the current cursor position
        # This is likely an indication of an opening function call
        # We need to determine the number of arguments that the function takes
        # This can be done by using the textDocument/completion request to obtain the signature at the current cursor position
        # And from signature, trivially return the number of arguments
        relative_file_path = self.monitor_file_buffer.file_path
        line, column = self.monitor_file_buffer.current_lc

        lsp_text = self.monitor_file_buffer.lsp.get_open_file_text(relative_file_path)
        request_idx = TextUtils.get_index_from_line_col(lsp_text, line, column)
        assert lsp_text[request_idx-1] == '('

        deleted_text = self.monitor_file_buffer.lsp.delete_text_between_positions(
            relative_file_path,
            Position(line=line, character=column-1),
            Position(line=line, character=column),
        )
        assert deleted_text == '('

        completions = await self.monitor_file_buffer.lsp.request_completions(relative_file_path, line, column-1)

        self.monitor_file_buffer.lsp.insert_text_at_position(
            relative_file_path,
            line=line,
            column=column-1,
            text_to_be_inserted='(',
        )

        # TODO: Handle the case of multiple overloaded methods
        if len(completions) != 1:
            return None
        
        signature = completions[0]["detail"]
        regex = r'.*\((.*)\) : .*'
        match = re.match(regex, signature)
        num_args = None
        if match is not None:
            args = match.group(1)
            if args == '':
                num_args = 0
            elif ',' not in args:
                num_args = 1
            else:
                num_args = len(args.split(','))

        return num_args

    def convert_nested_to_flat(self, s: str) -> str:
        """
        Converts a nested method call to a flat method call
        """
        depth = 0
        new_s = ""
        for c in s:
            if c == '(':
                depth += 1
                new_s += '.'
            elif c == ')':
                depth -= 1
                new_s += '.'
            elif c == ',':
                if depth == 0:
                    new_s += ','
                else:
                    new_s += '.'
            else:
                if c.strip() != '':
                    new_s += '.'
        return new_s
    
    def check_if_method_call_closed(self, s: str) -> bool:
        """
        Given a string of form "(...)", checks if the method call has been closed
        True for "(...)", "(...))", "(...)))", etc.
        False for "(...(", "(...(()", etc.
        """
        assert s[0] == '('
        depth = 1
        for c in s[1:]:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            
            if depth == 0:
                return True
        
        return False

    async def update(self, generated_token: str):
        """
        Updates the monitor state based on the generated token
        """
        for c in generated_token:
            self.monitor_file_buffer.append_text(c)
            await self.pre()

        current_text = self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path)
        to_remove_indices: List[int] = []
        for idx, open_method_call in enumerate(self.open_method_calls):
            # Check if the method_call has now been closed
            if self.check_if_method_call_closed(current_text[open_method_call.opening_idx:]):
                to_remove_indices.append(idx)
            
            view_for_this_method_call = self.convert_nested_to_flat(current_text[open_method_call.opening_idx+1:])
            assert len(view_for_this_method_call) != 0
            splits = view_for_this_method_call.split(',')
            if open_method_call.num_tot_args is None:
                continue
            num_args_left = open_method_call.num_tot_args - len(splits)
            # if len(splits[-1]) > 0:
            #     num_args_left -= 1
            open_method_call.num_args_left = num_args_left
        
        for idx in to_remove_indices[::-1]:
            self.open_method_calls.pop(idx)
