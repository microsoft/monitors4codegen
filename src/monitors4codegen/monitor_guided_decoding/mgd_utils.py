"""
This module provides the utility functions for handling programming language text
"""

import code_tokenize as ctok

from typing import List
from monitors4codegen.multilspy.multilspy_config import Language


class PLUtils:
    """
    This class provides various utility functions for handling programming language text
    """

    @staticmethod
    def tokenizer_pl(inp_text: str, lang: Language) -> List[ctok.tokens.ASTToken]:
        """
        Tokenizes the given text using code_tokenize
        """
        lang_s = str(lang) if lang != Language.CSHARP else "c-sharp"
        if inp_text.strip() == "":
            return []
        lsp_text_lang_tokenized: List[ctok.tokens.ASTToken] = ctok.tokenize(
            inp_text, lang=lang_s, syntax_error="ignore"
        )
        lsp_text_lang_tokenized: List[ctok.tokens.ASTToken] = [tok for tok in lsp_text_lang_tokenized if tok.text != ""]
        return lsp_text_lang_tokenized

    @staticmethod
    def get_opening_bracket_stream(inp_text: str, lang: Language) -> List[str]:
        """
        Returns the list of opened brackets in the given text
        """
        bracket_stream: List[str] = []
        err = False
        lsp_text_lang_tokenized = PLUtils.tokenizer_pl(inp_text, lang)
        for tok in lsp_text_lang_tokenized:
            if tok.type in ["{", "(", "["]:
                bracket_stream.append(tok.type)
            elif tok.type in ["}", ")", "]"]:
                if len(bracket_stream) == 0:
                    err = True
                    break
                if (
                    (tok.type == "}" and bracket_stream[-1] == "{")
                    or (tok.type == ")" and bracket_stream[-1] == "(")
                    or (tok.type == "]" and bracket_stream[-1] == "[")
                ):
                    bracket_stream.pop()
                else:
                    err = True
                    break
        if err:
            raise Exception("Invalid bracket stream")
        return bracket_stream

    @staticmethod
    def get_closing_bracket_stream(inp_text: str, lang: Language) -> List[str]:
        """
        Returns the list of closing brackets in the given text
        """
        bracket_stream: List[str] = []
        err = False
        lsp_text_lang_tokenized = PLUtils.tokenizer_pl(inp_text, lang)
        for tok in lsp_text_lang_tokenized[::-1]:
            if tok.type in ["}", ")", "]"]:
                bracket_stream.append(tok.type)
            elif tok.type in ["{", "(", "["]:
                if len(bracket_stream) == 0:
                    err = True
                    break
                if (
                    (tok.type == "{" and bracket_stream[-1] == "}")
                    or (tok.type == "(" and bracket_stream[-1] == ")")
                    or (tok.type == "[" and bracket_stream[-1] == "]")
                ):
                    bracket_stream.pop()
                else:
                    err = True
                    break
        if err:
            raise Exception("Invalid bracket stream")
        return bracket_stream[::-1]
