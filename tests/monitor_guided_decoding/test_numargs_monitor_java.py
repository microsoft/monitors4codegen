"""
This file contains tests for Monitor-Guided Decoding for correct number of arguments in Java
"""

import torch
import transformers
import pytest

from pathlib import PurePath
from monitors4codegen.multilspy.language_server import SyncLanguageServer
from monitors4codegen.multilspy.multilspy_config import Language
from tests.test_utils import create_test_context, is_cuda_available
from transformers import AutoTokenizer, AutoModelForCausalLM
from monitors4codegen.multilspy.multilspy_utils import TextUtils
from monitors4codegen.monitor_guided_decoding.monitors.numargs_monitor import NumMethodArgumentsMonitor
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.hf_gen import MGDLogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from monitors4codegen.multilspy.multilspy_types import Position
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import HFTokenizerWrapper

pytest_plugins = ("pytest_asyncio",)

@pytest.mark.asyncio
async def test_multilspy_java_clickhouse_highlevel_sinker_modified_numargs():
    """
    Test the working of numargs_monitor with Java repository - clickhouse-highlevel-sinker modified
    """
    code_language = Language.JAVA
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/LakshyAAAgrawal/clickhouse-highlevel-sinker/",
        "repo_commit": "5775fd7a67e7b60998e1614cf44a8a1fc3190ab0"
    }

    device = torch.device('cuda' if is_cuda_available() else 'cpu')

    model: transformers.modeling_utils.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "bigcode/santacoder", trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")

    with create_test_context(params) as context:
        lsp = SyncLanguageServer.create(context.config, context.logger, context.source_directory)
        # All the communication with the language server must be performed inside the context manager
        # The server process is started when the context manager is entered and is terminated when the context manager is exited.
        # The context manager is an asynchronous context manager, so it must be used with async with.
        with lsp.start_server():
            completions_filepath = "src/main/java/com/xlvchao/clickhouse/datasource/ClickHouseDataSource.java"
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=75, character=28),
                    Position(line=77, character=4)
                )
                assert deleted_text == """arr[0], Integer.parseInt(arr[1]))
                .build();
    """

                prompt_pos = (75, 28)

                with open(str(PurePath(context.source_directory, completions_filepath)), "r") as f:
                    filecontent = f.read()

                pos_idx = TextUtils.get_index_from_line_col(filecontent, prompt_pos[0], prompt_pos[1])
                assert filecontent[pos_idx] == "a"
                prompt = filecontent[:pos_idx]
                assert prompt[-1] == "("
                prompt_tokenized = tokenizer.encode(prompt, return_tensors="pt").cuda()[:, -(2048 - 512) :]

                gen = model.generate(
                    prompt_tokenized, do_sample=False, max_new_tokens=30, early_stopping=True
                )
                generated_code_without_mgd = tokenizer.decode(gen[0, -30:])

                assert (
                    generated_code_without_mgd ==
                    "arr[0])\n               .withPort(Integer.parseInt(arr[1]))\n               .build();\n    }\n\n    private List<ServerNode>"
                )

                filebuffer = MonitorFileBuffer(
                    lsp.language_server,
                    completions_filepath,
                    prompt_pos,
                    prompt_pos,
                    code_language,
                )
                monitor = NumMethodArgumentsMonitor(HFTokenizerWrapper(tokenizer), filebuffer)
                mgd_logits_processor = MGDLogitsProcessor([monitor], lsp.language_server.server.loop)

                # Generate code using santacoder model with the MGD logits processor and greedy decoding
                logits_processor = LogitsProcessorList([mgd_logits_processor])
                gen = model.generate(
                    prompt_tokenized,
                    do_sample=False,
                    max_new_tokens=30,
                    logits_processor=logits_processor,
                    early_stopping=True,
                )

                generated_code = tokenizer.decode(gen[0, -30:])

                assert (
                    generated_code ==
                    "arr[0].trim(), Integer.parseInt(arr[1].trim()))\n               .build();\n    }\n\n    private List<ServerNode> convertTo"
                )
