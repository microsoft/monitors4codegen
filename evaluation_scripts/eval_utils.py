from typing import List, Tuple
from pygments.lexers import JavaLexer
from pygments.token import Token

def find_method_stop_actual(gen_text: str) -> Tuple[str, int]:
    assert gen_text[0] == '{', gen_text
    jlex = JavaLexer()
    toks = list(jlex.get_tokens(gen_text))
    balance = -1
    int_toks = []
    for tok in toks:
        int_toks.append(tok[1])
        if tok[1] == '{' and Token.Punctuation in tok[0].split():
            balance += 1
        elif tok[1] == '}' and Token.Punctuation in tok[0].split():
            balance -= 1
        
        if balance == -1:
            break
    return (''.join(int_toks), balance+1)

def find_method_stop(gen_text: str) -> str:
    """
    Given the completed text for a method, returns the text for the method
    removing all of the extra text beyond the matching bracket.

    For example, if the input is:
    ```
        int a = 0;
        int b = 1;
        if (a == b) {
            return a;
        }
        return b;
    }

    public int foo() {
        return 0;
    }
    ```

    then the output is:
    ```
        int a = 0;
        int b = 1;
        if (a == b) {
            return a;
        }
        return b;
    }
    ```
    """
    text, balance = find_method_stop_actual('{' + gen_text)
    text = text.rstrip()
    assert gen_text.startswith(text[1:]), (gen_text, text, balance)
    return text[1:] + ('}'*balance)

def get_identifiers(text: str) -> List[str]:
    """
    Returns the list of identifiers in the input text as per the PL tokenizer
    """
    if len(text.strip()) == 0:
        return []
    else:
        j = JavaLexer()
        ctok = list(j.get_tokens(text))
        l = []
        for tok in ctok:
            if Token.Name in tok[0].split() or tok[1] == 'class':
                l.append(tok[1])
        return l

def tokenizer_pl(text: str) -> List[str]:
    """
    Tokenizes the input text as per the PL tokenizer removing all whitespaces
    """
    if len(text.strip()) == 0:
        return []
    else:
        j = JavaLexer()
        ctok = list(j.get_tokens(text))
        l = []
        for tok in ctok:
            if Token.Text in tok[0].split() and tok[1].strip() != '':
                raise Exception(text, tok)
            elif Token.Text.Whitespace in tok[0].split():
                assert tok[1].strip() == '', (text, ctok, tok)
            else:
                l.append(tok[1])
        return l

def get_first_token(inp_text: str) -> str:
    """
    Returns the first token as per the PL tokenizer for the input text
    """
    inp_text = inp_text.lstrip()
    tokens = tokenizer_pl(inp_text)
    if len(tokens) == 0:
        return None
    return tokens[0]