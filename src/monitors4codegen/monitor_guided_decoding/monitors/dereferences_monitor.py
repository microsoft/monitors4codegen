"""
This module provides the instantiation of dereferences monitor
"""

import code_tokenize as ctok

from typing import List, Union, Set
from enum import Enum

from monitors4codegen.multilspy import multilspy_types
from monitors4codegen.multilspy.multilspy_config import Language
from monitors4codegen.multilspy.multilspy_utils import TextUtils
from monitors4codegen.multilspy.multilspy_types import Position

from monitors4codegen.monitor_guided_decoding.monitor import Monitor, MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import TokenizerWrapper
from monitors4codegen.monitor_guided_decoding.mgd_utils import PLUtils


class DecoderStates(Enum):
    """
    Enum for the state of the decoder
    """

    UnInitialized = 0
    S0 = 1
    Constrained = 2


class DereferencesMonitor(Monitor):
    """
    Provides the dereferences monitor
    """

    def __init__(self, tokenizer: TokenizerWrapper, monitor_file_buffer: MonitorFileBuffer, responsible_for_file_buffer_state: bool = True) -> None:
        super().__init__(tokenizer, monitor_file_buffer, responsible_for_file_buffer_state)
        self.decoder_state = DecoderStates.UnInitialized
        self.all_break_chars = DereferencesMonitor.find_all_break_chars(
            self.tokenizer.tokenizer_char_set, monitor_file_buffer.language
        )
        self.prompt_len: Union[None, int] = None
        self.legal_completions: Union[List[str], None] = None

    @staticmethod
    def find_all_break_chars(charset: Set[str], language: Language) -> Set[str]:
        """
        Finds the set of characters, which when appended to the end of an identifier, cause the identifier to be terminated
        For example "," is a breaking character, since "abc," is 2 tokens, "abc" and ","
        On the other hand, "a" is not a breaking character, since "abca" is a single token
        """
        all_break_chars: Set[str] = set()
        for vocab_char in charset:
            toks: List[ctok.tokens.ASTToken] = PLUtils.tokenizer_pl("abc" + vocab_char, language)
            toks = [t for t in toks if t.text != ""]
            if len(toks) == 0 or toks[0].text == "abc":
                all_break_chars.add(vocab_char)
        return all_break_chars

    async def initialize(self, input_ids: List[int], input_text: str) -> None:
        """
        Initializes the monitor when it is invoked for the first time with inputs
        """
        self.prompt_len = len(input_ids)
        await self.pre()

    async def pre(self) -> None:
        """
        Checks if the static analysis should be performed at this point.
        In case of dereferences monitor, the last character shuold be a dot
        """
        cursor_idx = TextUtils.get_index_from_line_col(
            self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path),
            self.monitor_file_buffer.current_lc[0],
            self.monitor_file_buffer.current_lc[1],
        )
        text_upto_cursor = self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path)[
            :cursor_idx
        ]
        if text_upto_cursor[-1] != ".":
            self.decoder_state = DecoderStates.S0
            return

        completions = await self.a_phi()

        if len(completions) == 0:
            self.decoder_state = DecoderStates.S0
        else:
            self.decoder_state = DecoderStates.Constrained
            self.legal_completions = completions

    async def maskgen(self, input_ids: List[int]) -> List[int]:
        """
        Takes the list of input tokens, and returns the list of tokens to be blacklisted in the next step
        
        maskgen is invoked for every new token to be generated
        The first time it is invoked, maskgen performs the initialization
        Subsequent invocations are handled based on the current state of the decoder
        """

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

        if self.decoder_state == DecoderStates.S0:
            # If the decoder is in state S0, then need to check pre()
            # pre() will determine if the decoder should transition to state S1
            # If so, it invokes a_phi() and transitions the decoder state
            await self.pre()

        if self.decoder_state == DecoderStates.Constrained:
            # If the decoder is in state S1, then generate the set of blacklisted tokens
            # based on the current state of the monitor, legal_completions
            possible_token_ids: Set[int] = set()
            for legal_suffix in self.legal_completions:
                # If a token contains part of the end of an identifier followed by some breaking characters, then we allow it
                # allow decoding of tokens like 'abc<', basically tokens that span identifier boundaries
                if self.tokenizer.vocab_trie.has_node(legal_suffix) != 0:
                    for suffix_token, suffix_token_id in self.tokenizer.vocab_trie.iteritems(prefix=legal_suffix):
                        if suffix_token[len(legal_suffix) : len(legal_suffix) + 1] in self.all_break_chars:
                            possible_token_ids.add(suffix_token_id)

                # If a token is a prefix of the remaining suffix, then we allow it
                for suffix_token, suffix_token_id in self.tokenizer.vocab_trie.prefixes(legal_suffix):
                    possible_token_ids.add(suffix_token_id)

            blacklisted_ids = [i for i in self.tokenizer.all_token_ids if i not in possible_token_ids]
        else:
            blacklisted_ids = []

        return blacklisted_ids

    async def a_phi(self) -> List[str]:
        """
        Uses multilspy to perform static analysis and returns the list of type-compliant dereferences
        at the current cursor position (which ends with a dot)
        """
        relative_file_path = self.monitor_file_buffer.file_path
        line, column = self.monitor_file_buffer.current_lc

        with self.monitor_file_buffer.lsp.open_file(relative_file_path):
            legal_completions1 = await self.monitor_file_buffer.lsp.request_completions(
                relative_file_path, line, column, allow_incomplete=True
            )
            legal_completions1 = [
                completion["completionText"]
                for completion in legal_completions1
                if completion["kind"] != multilspy_types.CompletionItemKind.Keyword
            ]
            lsp_text = self.monitor_file_buffer.lsp.get_open_file_text(relative_file_path)
            request_idx = TextUtils.get_index_from_line_col(lsp_text, line, column)
            opening_bracket_stream = PLUtils.get_opening_bracket_stream(
                lsp_text[:request_idx], self.monitor_file_buffer.language
            )
            if len(opening_bracket_stream) == 0:
                return legal_completions1

            closing_bracket_stream = PLUtils.get_closing_bracket_stream(
                lsp_text[request_idx:], self.monitor_file_buffer.language
            )
            if len(opening_bracket_stream) <= len(closing_bracket_stream):
                return legal_completions1

            err = False
            for j in range(len(closing_bracket_stream)):
                if closing_bracket_stream[-j - 1] == "}" and opening_bracket_stream[j] != "{":
                    err = True
                    break
                elif closing_bracket_stream[-j - 1] == ")" and opening_bracket_stream[j] != "(":
                    err = True
                    break
                elif closing_bracket_stream[-j - 1] == "]" and opening_bracket_stream[j] != "[":
                    err = True
                    break

            if err:
                return legal_completions1

            opening_bracket_stream = opening_bracket_stream[len(closing_bracket_stream) :]
            remaining_close_brackets = list(
                map(lambda x: "}" if x == "{" else (")" if x == "(" else "]"), opening_bracket_stream)
            )[::-1]

            insert_text = "".join(remaining_close_brackets)
            updated_cursor = self.monitor_file_buffer.lsp.insert_text_at_position(
                relative_file_path, line, column, insert_text
            )
            assert updated_cursor["line"] == line
            assert updated_cursor["character"] == column + len(insert_text)
            legal_completions2 = await self.monitor_file_buffer.lsp.request_completions(
                relative_file_path, line, column, allow_incomplete=True
            )
            legal_completions2 = [
                completion["completionText"]
                for completion in legal_completions2
                if completion["kind"] != multilspy_types.CompletionItemKind.Keyword
            ]

            deleted_text = self.monitor_file_buffer.lsp.delete_text_between_positions(
                relative_file_path,
                Position(line=line, character=column),
                Position(line=line, character=column + len(insert_text)),
            )
            assert deleted_text == insert_text

            return list(set(legal_completions1 + legal_completions2))

    async def update(self, generated_token: str):
        """
        Updates the monitor state based on the generated token
        """
        if self.responsible_for_file_buffer_state:
            self.monitor_file_buffer.append_text(generated_token)
        if self.decoder_state == DecoderStates.Constrained:
            for break_char in self.all_break_chars:
                if break_char in generated_token:
                    self.decoder_state = DecoderStates.S0
                    self.legal_completions = None
                    return

            # No breaking characters found. Continue in constrained state
            self.legal_completions = [
                legal_completion[len(generated_token) :]
                for legal_completion in self.legal_completions
                if legal_completion.startswith(generated_token)
            ]
        else:
            # Nothing to be done in other states
            return
