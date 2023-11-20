"""
This file contains tests for Monitor-Guided Decoding for dereferences in Java, with OpenAI models
"""
import os
import pytest
import tiktoken
import torch

from openai import OpenAI
from pathlib import PurePath
from monitors4codegen.multilspy.language_server import SyncLanguageServer
from monitors4codegen.multilspy.multilspy_config import Language
from tests.test_utils import create_test_context
from monitors4codegen.multilspy.multilspy_utils import TextUtils
from monitors4codegen.monitor_guided_decoding.monitors.dereferences_monitor import DereferencesMonitor
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.multilspy.multilspy_types import Position
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import TikTokenWrapper
from monitors4codegen.monitor_guided_decoding.openai_gen import openai_mgd, OpenAI_Models

@pytest.mark.asyncio
@pytest.mark.skipif(not "OPENAI_API_KEY" in os.environ, reason="OpenAI API key not found")
async def test_dereferences_monitor_java_openai_clickhouse_highlevel_sinker():
    """
    Test the working of dereferences monitor with Java repository - clickhouse-highlevel-sinker modified
    """
    code_language = Language.JAVA
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/Index103000/clickhouse-highlevel-sinker/",
        "repo_commit": "ee31d278918fe5e64669a6840c4d8fb53889e573",
    }

    client = OpenAI()
    client.api_key = os.environ["OPENAI_API_KEY"]

    encoding = tiktoken.encoding_for_model("text-davinci-003")
    tokenizer = TikTokenWrapper(encoding)

    with create_test_context(params) as context:
        lsp = SyncLanguageServer.create(context.config, context.logger, context.source_directory)
        completions_filepath = "src/main/java/com/xlvchao/clickhouse/datasource/ClickHouseDataSource.java"
        # All the communication with the language server must be performed inside the context manager
        # The server process is started when the context manager is entered and is terminated when the context manager is exited.
        with lsp.start_server():
            with lsp.open_file(completions_filepath):
                filebuffer = MonitorFileBuffer(lsp.language_server, completions_filepath, (74, 17), (74, 17), code_language, "")
                deleted_text = filebuffer.lsp.delete_text_between_positions(
                    completions_filepath, Position(line=74, character=17), Position(line=78, character=4)
                )
                assert (
                    deleted_text
                    == """newServerNode()
                .withIp(arr[0])
                .withPort(Integer.parseInt(arr[1]))
                .build();
    """
                )

                prompt_pos = (74, 17)

                with open(str(PurePath(context.source_directory, completions_filepath)), "r") as f:
                    filecontent = f.read()

                pos_idx = TextUtils.get_index_from_line_col(filecontent, prompt_pos[0], prompt_pos[1])
                assert filecontent[pos_idx] == "n"
                prompt = filecontent[:pos_idx]
                assert prompt[-1] == "."
                prompt_tokenized = encoding.encode(prompt)[-(2048 - 512):]

                completions = client.completions.create(
                    model="text-davinci-003",
                    prompt=tokenizer.decode(torch.Tensor(prompt_tokenized).long(), skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    max_tokens=30,
                    temperature=0
                )
                
                generated_code_without_mgd = completions.choices[0].text

                assert (
                    generated_code_without_mgd ==
                    'newBuilder()\n                .host(arr[0])\n                .port(Integer.parseInt(arr[1]))\n                .'
                )

                filebuffer = MonitorFileBuffer(
                    lsp.language_server,
                    completions_filepath,
                    prompt_pos,
                    prompt_pos,
                    code_language,
                )
                monitor = DereferencesMonitor(tokenizer, filebuffer)

                gen_tokens, generated_code = openai_mgd(
                    client=client,
                    model=OpenAI_Models.TD3,
                    tokenizer=tokenizer,
                    prompt_tokenized=prompt_tokenized,
                    temp=0,
                    top_p=0.95,
                    monitor=monitor,
                    num_new_tokens=30,
                )

                assert (
                    generated_code ==
                    'newServerNode()\n                .withIp(arr[0])\n                .withPort(Integer.parseInt(arr[1]'
                )

@pytest.mark.asyncio
@pytest.mark.skipif(not "OPENAI_API_KEY" in os.environ, reason="OpenAI API key not found")
async def test_dereferences_monitor_java_openai_clickhouse_highlevel_sinker_modified():
    """
    Test the working of dereferences monitor with Java repository - clickhouse-highlevel-sinker modified
    """
    code_language = Language.JAVA
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/LakshyAAAgrawal/clickhouse-highlevel-sinker/",
        "repo_commit": "5775fd7a67e7b60998e1614cf44a8a1fc3190ab0"
    }

    client = OpenAI()
    client.api_key = os.environ["OPENAI_API_KEY"]

    encoding = tiktoken.encoding_for_model("text-davinci-003")
    tokenizer = TikTokenWrapper(encoding)

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
                    Position(line=75, character=17),
                    Position(line=77, character=4)
                )
                assert deleted_text == """withIpPort(arr[0], Integer.parseInt(arr[1]))
                .build();
    """

                prompt_pos = (75, 17)

                with open(str(PurePath(context.source_directory, completions_filepath)), "r") as f:
                    filecontent = f.read()

                pos_idx = TextUtils.get_index_from_line_col(filecontent, prompt_pos[0], prompt_pos[1])
                assert filecontent[pos_idx] == "w"
                prompt = filecontent[:pos_idx]
                assert prompt[-1] == "."
                prompt_tokenized = encoding.encode(prompt)[-(2048 - 512):]

                completions = client.completions.create(
                    model="text-davinci-003",
                    prompt=tokenizer.decode(torch.Tensor(prompt_tokenized).long(), skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    max_tokens=30,
                    temperature=0
                )
                
                generated_code_without_mgd = completions.choices[0].text

                assert (
                    generated_code_without_mgd ==
                    'host(arr[0])\n                .port(Integer.parseInt(arr[1]))\n                .build();\n    }\n\n'
                )

                filebuffer = MonitorFileBuffer(
                    lsp.language_server,
                    completions_filepath,
                    prompt_pos,
                    prompt_pos,
                    code_language,
                )
                monitor = DereferencesMonitor(tokenizer, filebuffer)

                gen_tokens, generated_code = openai_mgd(
                    client=client,
                    model=OpenAI_Models.TD3,
                    tokenizer=tokenizer,
                    prompt_tokenized=prompt_tokenized,
                    temp=0,
                    top_p=0.95,
                    monitor=monitor,
                    num_new_tokens=30,
                )

                assert (
                    generated_code ==
                    'withIpPort(arr[0], Integer.parseInt(arr[1]))\n                .build();\n    }\n\n    private'
                )
