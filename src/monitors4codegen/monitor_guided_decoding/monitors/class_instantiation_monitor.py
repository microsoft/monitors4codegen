"""
This module provides the class-instantiation monitor, that is invoked when "new " is typed to instantiate new classes
"""

import os

from pathlib import PurePath
from typing import List
from monitors4codegen.monitor_guided_decoding.monitors.dereferences_monitor import DereferencesMonitor, DecoderStates
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import TokenizerWrapper
from monitors4codegen.multilspy.multilspy_utils import TextUtils, FileUtils
from monitors4codegen.multilspy import multilspy_types

class ClassInstantiationMonitor(DereferencesMonitor):
    """
    Class Instantiation Monitor that is invoked when "new " is typed to instantiate new classes
    """

    def __init__(self, tokenizer: TokenizerWrapper, monitor_file_buffer: MonitorFileBuffer, responsible_for_file_buffer_state: bool = True) -> None:
        super().__init__(tokenizer, monitor_file_buffer, responsible_for_file_buffer_state)

    async def pre(self) -> None:
        cursor_idx = TextUtils.get_index_from_line_col(
            self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path),
            self.monitor_file_buffer.current_lc[0],
            self.monitor_file_buffer.current_lc[1],
        )
        text_upto_cursor = self.monitor_file_buffer.lsp.get_open_file_text(self.monitor_file_buffer.file_path)[
            :cursor_idx
        ]

        # TODO: pre can be improved by checking for "new", and obtaining completions, and then prefixing a whitespace
        if not text_upto_cursor.endswith("new "):
            self.decoder_state = DecoderStates.S0
            return

        completions = await self.a_phi()
        if len(completions) == 0:
            self.decoder_state = DecoderStates.S0
        else:
            self.decoder_state = DecoderStates.Constrained
            self.legal_completions = completions
    
    async def a_phi(self) -> List[str]:
        """
        Find the set of classes in the repository
        Filter out the set of abstract classes in the repository
        Remaining classes are instantiable. Return their names as legal completions
        """

        legal_completions: List[str] = []
        repository_root_path = self.monitor_file_buffer.lsp.repository_root_path
        for path, _, files in os.walk(repository_root_path):
            for file in files:
                if file.endswith(".java"):
                    filecontents = FileUtils.read_file(self.monitor_file_buffer.lsp.logger, str(PurePath(path, file)))
                    relative_file_path = str(PurePath(os.path.relpath(str(PurePath(path, file)), repository_root_path)))
                    document_symbols, _ = await self.monitor_file_buffer.lsp.request_document_symbols(relative_file_path)
                    for symbol in document_symbols:
                        if symbol["kind"] != multilspy_types.SymbolKind.Class:
                            continue
                        decl_start_idx = TextUtils.get_index_from_line_col(filecontents, symbol["range"]["start"]["line"], symbol["range"]["start"]["character"])
                        decl_end_idx = TextUtils.get_index_from_line_col(filecontents, symbol["selectionRange"]["end"]["line"], symbol["selectionRange"]["end"]["character"])
                        decl_text = filecontents[decl_start_idx:decl_end_idx]
                        if "abstract" not in decl_text:
                            legal_completions.append(symbol["name"])

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