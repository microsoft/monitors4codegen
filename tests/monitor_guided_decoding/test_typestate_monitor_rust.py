"""
This file contains tests for MGD for typestate
"""

import pytest
import torch
import transformers

from pathlib import PurePath
from monitors4codegen.multilspy.language_server import SyncLanguageServer
from monitors4codegen.multilspy.multilspy_config import Language
from tests.test_utils import create_test_context, is_cuda_available
from transformers import AutoTokenizer, AutoModelForCausalLM
from monitors4codegen.multilspy.multilspy_utils import TextUtils
from monitors4codegen.monitor_guided_decoding.monitors.dereferences_monitor import DereferencesMonitor
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.hf_gen import MGDLogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from monitors4codegen.multilspy.multilspy_types import Position
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import HFTokenizerWrapper

pytest_plugins = ("pytest_asyncio",)

@pytest.mark.asyncio
async def test_typestate_monitor_rust_huggingface_models_mediaplayer() -> None:
    """
    Test the working of typestate monitor with Rust repository - mediaplayer
    """
    code_language = Language.RUST
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/LakshyAAAgrawal/MediaPlayer_example/",
        "repo_commit": "80cd910cfeb2a05c9e74b69773373c077b00b4c2",
    }

    device = torch.device('cuda' if is_cuda_available() else 'cpu')

    model: transformers.modeling_utils.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "bigcode/santacoder", trust_remote_code=True
    ).to(device) #
    tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")

    with create_test_context(params) as context:
        lsp = SyncLanguageServer.create(context.config, context.logger, context.source_directory)
        filepath = "src/playlist.rs"
        # All the communication with the language server must be performed inside the context manager
        # The server process is started when the context manager is entered and is terminated when the context manager is exited.
        with lsp.start_server():
            with lsp.open_file(filepath):
                filebuffer = MonitorFileBuffer(lsp.language_server, filepath, (10, 40), (10, 40), code_language, "")
                deleted_text = filebuffer.lsp.delete_text_between_positions(
                    filepath, Position(line=10, character=40), Position(line=12, character=4)
                )
                assert (
                    deleted_text
                    == """reset();
        media_player1 = media_player;
    """
                )
                monitor = DereferencesMonitor(HFTokenizerWrapper(tokenizer), filebuffer)
                mgd_logits_processor = MGDLogitsProcessor([monitor], lsp.language_server.server.loop)

                with open(str(PurePath(context.source_directory, filepath)), "r") as f:
                    filecontent = f.read()

                pos_idx = TextUtils.get_index_from_line_col(filecontent, 10, 40)
                assert filecontent[pos_idx] == "r"

                with open(str(PurePath(context.source_directory, "src/media_player.rs")), "r") as f:
                    classExprTypes = f.read()

                prompt = filecontent[:pos_idx] + """<fim-suffix>();
        media_player1 = media_player;
    }
}<fim-middle>."""
                assert prompt[-1] == "."
                prompt_tokenized = tokenizer.encode("<fim-prefix>" + classExprTypes + '\n' + prompt, return_tensors="pt")[:, -(2048-512):].to(device)

                generated_code_without_mgd = model.generate(
                    prompt_tokenized, do_sample=False, max_new_tokens=5, top_p=0.95, temperature=0.2,
                )

                num_gen_tokens = generated_code_without_mgd.shape[-1] - prompt_tokenized.shape[-1]
                generated_code_without_mgd = tokenizer.decode(generated_code_without_mgd[0, -num_gen_tokens:])

                assert (
                    generated_code_without_mgd
                    == """stop();
        let media"""
                )

                # Generate code using santacoder model with the MGD logits processor and greedy decoding
                logits_processor = LogitsProcessorList([mgd_logits_processor])
                generated_code = model.generate(
                    prompt_tokenized,
                    do_sample=False,
                    max_new_tokens=30,
                    logits_processor=logits_processor,
                    early_stopping=True,
                )

                num_gen_tokens = generated_code.shape[-1] - prompt_tokenized.shape[-1]
                generated_code = tokenizer.decode(generated_code[0, -num_gen_tokens:])

                assert (
                    generated_code
                    == """reset();
        let media_player = media_player.set_media_file_path(song);
        let media_player = media_"""
                )
