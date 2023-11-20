"""
Provides the definition of a monitor as per the Monitor-Guided Decoding framework
"""

import asyncio
import torch

from asyncio.events import AbstractEventLoop
from typing import List, Union
from transformers import LogitsProcessor
from monitors4codegen.monitor_guided_decoding.monitor import Monitor

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
