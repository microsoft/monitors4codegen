import pdb
import re
import json
from monitors4codegen.multilspy import SyncLanguageServer
from monitors4codegen.multilspy.multilspy_config import MultilspyConfig
from monitors4codegen.multilspy.multilspy_logger import MultilspyLogger

config = MultilspyConfig.from_dict({"code_language": "java"}) # Also supports "python", "rust", "csharp"
logger = MultilspyLogger()
lsp = SyncLanguageServer.create(config, logger, "/home/ken/r4e-small-test/")
with lsp.start_server():
    while True:
#        user_line = input("enter: line, column: ")
#        user_line = re.sub(r"\s+", "", user_line)
        #source_file="r4e-address/src/main/java/com/reputation/address/config/MainConfig.java"
        source_file = "r4e-address/src/main/java/com/reputation/address/service/AddressService.java"
#        [line_number, column_number] = user_line.split(",")
#        print(source_file, line_number, column_number)
#        result = lsp.request_definition(source_file, line_number, column_number)
        result = lsp.request_code_action(source_file, 99999, 99999)
        #result = lsp.request_document_symbols(source_file)
        #result = lsp.request_hover(source_file, line_number, column_number)
        print(json.dumps(result))
        exit()
    # result2 = lsp.request_completions(
    #     ...
    # )
    # result3 = lsp.request_references(
    #     ...
    # )
    # result4 = lsp.request_document_symbols(
    #     ...
    # )
    # result5 = lsp.request_hover(
    #     ...
    # )
