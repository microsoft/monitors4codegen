"""
Provides the definition of a monitor as per the Monitor-Guided Decoding framework
"""

from typing import List, Tuple
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import TokenizerWrapper
from monitors4codegen.multilspy import LanguageServer
from monitors4codegen.multilspy.multilspy_config import Language
from dataclasses import dataclass
from monitors4codegen.multilspy.multilspy_utils import TextUtils

@dataclass
class MonitorFileBuffer:
    """
    Dataclass for storing the state of the monitor for the prompt file in which the generation is happening
    """

    lsp: LanguageServer
    file_path: str
    prompt_lc: Tuple[int, int]
    current_lc: Tuple[int, int]
    language: Language
    gen_text: str = ""

    def append_text(self, text: str):
        """
        Appends the given text to the prompt file and returns the new line and character
        """
        current_lc_index = TextUtils.get_index_from_line_col(
            self.lsp.get_open_file_text(self.file_path), self.current_lc[0], self.current_lc[1]
        )
        new_lc = self.lsp.insert_text_at_position(self.file_path, self.current_lc[0], self.current_lc[1], text)
        self.current_lc = (new_lc["line"], new_lc["character"])
        self.gen_text += text
        assert current_lc_index + len(text) == TextUtils.get_index_from_line_col(
            self.lsp.get_open_file_text(self.file_path), self.current_lc[0], self.current_lc[1]
        )


class Monitor:
    """
    Provides the definition of a monitor as per the Monitor-Guided Decoding framework
    """

    def __init__(self, tokenizer: TokenizerWrapper, monitor_file_buffer: MonitorFileBuffer, responsible_for_file_buffer_state: bool = True) -> None:
        self.tokenizer = tokenizer
        self.monitor_file_buffer = monitor_file_buffer
        self.responsible_for_file_buffer_state = responsible_for_file_buffer_state

    async def pre(self) -> None:
        """
        If the current state is uninitialized, or s0, this function checks
        if the static analysis should be performed at this point.
        If yes, it invokes the static analysis, and updates the state.
        """
        raise NotImplementedError()

    async def maskgen(self, input_ids: List[int]) -> List[int]:
        """
        Given input_ids, which is the list of token ids generated so far (or input for the first time),
        this function returns the list of token ids that should be masked for the next token generation.

        This is the function that is invoked by the end user at every token decodes.
        """
        raise NotImplementedError()

    def a_phi(self):
        """
        This function defines the implementation of the static analysis,
        and returns the result of the static analysis.
        It is invoked primarily by pre()
        """
        raise NotImplementedError()

    def update(self, generated_token: str):
        """
        This function updates the state of the monitor, given the generated token.
        """
        raise NotImplementedError()
