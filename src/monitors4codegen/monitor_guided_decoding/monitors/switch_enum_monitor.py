"""
This module provides the switch-enum monitor, that is invoked when "case " is typed in a switch statement
"""

from typing import List
from monitors4codegen.monitor_guided_decoding.monitors.dereferences_monitor import DereferencesMonitor, DecoderStates
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import TokenizerWrapper
from monitors4codegen.multilspy.multilspy_utils import TextUtils
from monitors4codegen.multilspy import multilspy_types

class SwitchEnumMonitor(DereferencesMonitor):
    """
    Provides the switch-enum monitor, that is invoked when "case " is typed in a switch statement to provide
    enum values as completions
    """
    def __init__(self, tokenizer: TokenizerWrapper, monitor_file_buffer: MonitorFileBuffer, responsible_for_file_buffer_state: bool = True) -> None:
        super().__init__(tokenizer, monitor_file_buffer, responsible_for_file_buffer_state)
        self.all_break_chars.remove('.')

    async def pre(self) -> None:
        cursor_idx = TextUtils.get_index_from_line_col(
            self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path),
            self.monitor_file_buffer.current_lc[0],
            self.monitor_file_buffer.current_lc[1],
        )
        text_upto_cursor = self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path)[
            :cursor_idx
        ]

        # TODO: pre can be improved by checking for r"switch.*case", and obtaining completions, and then prefixing a whitespace
        if not text_upto_cursor.endswith("case "):
            self.decoder_state = DecoderStates.S0
            return

        completions = await self.a_phi()
        if len(completions) == 0:
            self.decoder_state = DecoderStates.S0
        else:
            self.decoder_state = DecoderStates.Constrained
            self.legal_completions = completions
    
    async def a_phi(self) -> List[str]:
        relative_file_path = self.monitor_file_buffer.file_path
        line, column = self.monitor_file_buffer.current_lc

        with self.monitor_file_buffer.lsp.open_file(relative_file_path):
            legal_completions = await self.monitor_file_buffer.lsp.request_completions(
                relative_file_path, line, column
            )
            legal_completions = [
                completion["completionText"]
                for completion in legal_completions
                if completion["kind"] == multilspy_types.CompletionItemKind.EnumMember
            ]
            
            return legal_completions
    
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