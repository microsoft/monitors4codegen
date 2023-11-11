"""
Provides the definition of a monitor as per the Monitor-Guided Decoding framework
"""

import asyncio
import torch

from asyncio.events import AbstractEventLoop
from typing import List, Tuple, Union
from transformers import LogitsProcessor
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


class MGDLogitsProcessor(LogitsProcessor):
    """
    Provides the logits processor for monitor guided decoding
    """

    loop: AbstractEventLoop

    def __init__(self, monitors: List[Monitor], loop: Union[None, AbstractEventLoop] = None) -> None:
        super().__init__()

        if loop is None:
            self.loop = asyncio.get_event_loop()
        else:
            self.loop = loop

        self.monitors: List[Monitor] = monitors

    async def process_scores_for_single_input_id(
        self, segment_idx: int, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Asynchronously processes the scores for a single input id using the MGD framework
        """
        blacklisted_ids: List[int] = await self.monitors[segment_idx].maskgen(input_ids.tolist())
        output_scores: torch.FloatTensor = torch.where(
            torch.tensor([True if i in blacklisted_ids else False for i in range(scores.shape[0])]).to(scores.device),
            float("-inf") * torch.ones(scores.shape[0]).to(scores.device),
            scores,
        ).to(scores)
        return output_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        This method is called by the HuggingFace decoder, for every token generation with
        the input_ids (seen so far including prompt) and scores (for the next token).
        This method processes the scores using the MGD framework.
        """
        assert len(input_ids.shape) == 2
        assert input_ids.shape[0] == len(self.monitors)
        assert len(scores.shape) == 2

        async def f(input_ids_arg: torch.LongTensor, scores_arg: torch.FloatTensor):
            new_score_coroutines = [
                self.process_scores_for_single_input_id(i, input_ids_arg[i], scores_arg[i])
                for i in range(input_ids_arg.shape[0])
            ]
            new_scores = await asyncio.gather(*new_score_coroutines)
            return tuple(new_scores)

        future = asyncio.run_coroutine_threadsafe(f(input_ids, scores), self.loop)
        results = future.result()
        new_scores = torch.stack(results, dim=0).to(scores)
        return new_scores
