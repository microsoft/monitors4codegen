"""
This file contains tests for running the Java Language Server: Eclipse JDT.LS
"""

import pytest
from pathlib import PurePath
from monitors4codegen.multilspy import LanguageServer
from monitors4codegen.multilspy.multilspy_config import Language
from monitors4codegen.multilspy.multilspy_types import Position, CompletionItemKind
from tests.test_utils import create_test_context

pytest_plugins = ("pytest_asyncio",)

@pytest.mark.asyncio
async def test_multilspy_java_clickhouse_highlevel_sinker():
    """
    Test the working of multilspy with Java repository - clickhouse-highlevel-sinker
    """
    code_language = Language.JAVA
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/Index103000/clickhouse-highlevel-sinker/",
        "repo_commit": "ee31d278918fe5e64669a6840c4d8fb53889e573"
    }
    with create_test_context(params) as context:
        lsp = LanguageServer.create(context.config, context.logger, context.source_directory)

        # All the communication with the language server must be performed inside the context manager
        # The server process is started when the context manager is entered and is terminated when the context manager is exited.
        # The context manager is an asynchronous context manager, so it must be used with async with.
        async with lsp.start_server():
            filepath = str(PurePath("src/main/java/com/xlvchao/clickhouse/component/ClickHouseSinkManager.java"))
            result = await lsp.request_definition(filepath, 44, 59)

            assert isinstance(result, list)
            assert len(result) == 1
            item = result[0]
            assert item["relativePath"] == str(
                PurePath("src/main/java/com/xlvchao/clickhouse/component/ScheduledCheckerAndCleaner.java")
            )
            assert item["range"] == {
                "start": {"line": 22, "character": 11},
                "end": {"line": 22, "character": 37},
            }

            # TODO: The following test is running flaky on Windows. Investigate and fix.
            # On Windows, it returns the correct result sometimes and sometimes it returns the following:
            # incorrect_output = [
            #     {
            #         "range": {"end": {"character": 86, "line": 24}, "start": {"character": 65, "line": 24}},
            #         "relativePath": "src\\main\\java\\com\\xlvchao\\clickhouse\\component\\ClickHouseSinkManager.java",
            #     },
            #     {
            #         "range": {"end": {"character": 61, "line": 2}, "start": {"character": 7, "line": 2}},
            #         "relativePath": "src\\test\\java\\com\\xlvchao\\clickhouse\\SpringbootDemo.java",
            #     },
            #     {
            #         "range": {"end": {"character": 29, "line": 28}, "start": {"character": 8, "line": 28}},
            #         "relativePath": "src\\test\\java\\com\\xlvchao\\clickhouse\\SpringbootDemo.java",
            #     },
            #     {
            #         "range": {"end": {"character": 69, "line": 28}, "start": {"character": 48, "line": 28}},
            #         "relativePath": "src\\test\\java\\com\\xlvchao\\clickhouse\\SpringbootDemo.java",
            #     },
            # ]

            result = await lsp.request_references(filepath, 82, 27)

            assert isinstance(result, list)
            assert len(result) == 2

            for item in result:
                del item["uri"]
                del item["absolutePath"]

            assert result == [
                {
                    "relativePath": str(
                        PurePath("src/main/java/com/xlvchao/clickhouse/component/ClickHouseSinkManager.java")
                    ),
                    "range": {
                        "start": {"line": 75, "character": 66},
                        "end": {"line": 75, "character": 85},
                    },
                },
                {
                    "relativePath": str(
                        PurePath("src/main/java/com/xlvchao/clickhouse/component/ClickHouseSinkManager.java")
                    ),
                    "range": {
                        "start": {"line": 71, "character": 12},
                        "end": {"line": 71, "character": 31},
                    },
                },
            ]

            completions_filepath = "src/main/java/com/xlvchao/clickhouse/datasource/ClickHouseDataSource.java"
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=74, character=17),
                    Position(line=78, character=4)
                )
                assert deleted_text == """newServerNode()
                .withIp(arr[0])
                .withPort(Integer.parseInt(arr[1]))
                .build();
    """
                completions = await lsp.request_completions(completions_filepath, 74, 17)
                completions = [completion["completionText"] for completion in completions]
                assert set(completions) == set(['class', 'newServerNode'])
            
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=75, character=17),
                    Position(line=78, character=4)
                )
                assert deleted_text == """withIp(arr[0])
                .withPort(Integer.parseInt(arr[1]))
                .build();
    """
                completions = await lsp.request_completions(completions_filepath, 75, 17)
                completions = [completion["completionText"] for completion in completions]
                assert set(completions) == set(['build', 'equals', 'getClass', 'hashCode', 'toString', 'withIp', 'withPort', 'notify', 'notifyAll', 'wait', 'wait', 'wait', 'newServerNode'])
            
            completions_filepath = "src/main/java/com/xlvchao/clickhouse/component/ClickHouseSinkBuffer.java"
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=136, character=23),
                    Position(line=143, character=8)
                )
                assert deleted_text == """ClickHouseSinkBuffer(
                    this.writer,
                    this.writeTimeout,
                    this.batchSize,
                    this.clazz,
                    this.futures
            );
        """
                completions = await lsp.request_completions(completions_filepath, 136, 23)
                completions = [completion["completionText"] for completion in completions if completion["kind"] == CompletionItemKind.Constructor]
                assert completions == ['ClickHouseSinkBuffer']

@pytest.mark.asyncio
async def test_multilspy_java_clickhouse_highlevel_sinker_modified():
    """
    Test the working of multilspy with Java repository - clickhouse-highlevel-sinker
    """
    code_language = Language.JAVA
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/LakshyAAAgrawal/clickhouse-highlevel-sinker/",
        "repo_commit": "5775fd7a67e7b60998e1614cf44a8a1fc3190ab0"
    }
    with create_test_context(params) as context:
        lsp = LanguageServer.create(context.config, context.logger, context.source_directory)
        # All the communication with the language server must be performed inside the context manager
        # The server process is started when the context manager is entered and is terminated when the context manager is exited.
        # The context manager is an asynchronous context manager, so it must be used with async with.
        async with lsp.start_server():
            completions_filepath = "src/main/java/com/xlvchao/clickhouse/datasource/ClickHouseDataSource.java"
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=74, character=17),
                    Position(line=77, character=4)
                )
                assert deleted_text == """newServerNode()
                .withIpPort(arr[0], Integer.parseInt(arr[1]))
                .build();
    """
                completions = await lsp.request_completions(completions_filepath, 74, 17)
                completions = [completion["completionText"] for completion in completions]
                assert set(completions) == set(['class', 'newServerNode'])

            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=75, character=17),
                    Position(line=77, character=4)
                )
                assert deleted_text == """withIpPort(arr[0], Integer.parseInt(arr[1]))
                .build();
    """
                completions = await lsp.request_completions(completions_filepath, 75, 17)
                completions = [completion["completionText"] for completion in completions]
                assert set(completions) == set(['build', 'equals', 'getClass', 'hashCode', 'toString', 'withIpPort', 'notify', 'notifyAll', 'wait', 'wait', 'wait', 'newServerNode'])

            completions_filepath = "src/main/java/com/xlvchao/clickhouse/component/ClickHouseSinkBuffer.java"
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=136, character=23),
                    Position(line=143, character=8)
                )
                assert deleted_text == """ClickHouseSinkBuffer(
                    this.writer,
                    this.writeTimeout,
                    this.batchSize,
                    this.clazz,
                    this.futures
            );
        """
                completions = await lsp.request_completions(completions_filepath, 136, 23)
                completions = [completion["completionText"] for completion in completions if completion["kind"] == CompletionItemKind.Constructor]
                assert completions == ['ClickHouseSinkBuffer']
