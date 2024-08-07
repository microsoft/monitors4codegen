# Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context
Alternative title: Guiding Language Models of Code with Global Context using Monitors

## Introduction
This repository hosts the official code and data artifact for the paper ["Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context"](https://neurips.cc/virtual/2023/poster/70362) appearing at NeurIPS 2023 (["Guiding Language Models of Code with Global Context using Monitors"](https://arxiv.org/abs/2306.10763) on Arxiv). The work introduces Monitor-Guided Decoding (MGD) for code generation using Language Models, where a monitor uses static analysis to guide the decoding.

## Repository Contents
1. [Datasets](#1-datasets): PragmaticCode and DotPrompts
2. [Evaluation scripts](#2-evaluation-scripts): Scripts to evaluate LMs by taking as input inferences (code generated by the model) for examples in DotPrompts and producing score@k scores for the metrics reported in the paper: Compilation Rate (CR), Next-Identifier Match (NIM), Identifier-Sequence Match (ISM) and Prefix Match (PM).
3. [Inference Results over DotPrompts](#3-inference-results-over-dotprompts): Generated code for examples in DotPrompts with various model configurations reported in the paper. The graphs and tables reported in the paper can be reproduced by running the evaluation scripts on the provided inference results.
4. [`multilspy`](#4-multilspy): A cross-platform library designed to simplify the process of creating language server clients to query and obtain results of various static analyses from a wide variety of language servers that communicate over the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/). `multilspy` is intended to be used as a library to easily query various language servers, without having to worry about setting up their configurations and implementing the client-side of language server protocol. `multilspy` currently supports running language servers for Java, Rust, C# and Python, and we aim to expand this list with the help of the community.
5. [Monitor-Guided Decoding](#5-monitor-guided-decoding): Implementation of various monitors monitoring for different properties reported in the paper (for example: monitoring for type-valid identifier dereferences, monitoring for correct number of arguments to method calls, monitoring for typestate validity of method call sequences, etc.), spanning 3 programming languages.

**The `multilspy` library has now been migrated to [microsoft/multilspy](https://github.com/microsoft/multilspy).**

## Monitor-Guided Decoding: Motivating Example
For example, consider the partial code to be completed in the figure below. To complete this code, an LM has to generate identifiers consistent with the type of the object returned by `ServerNode.Builder.newServerNode()`. The method `newServerNode` and its return type, class `ServerNode.Builder`, are defined in another file. If an LM does not have information about the `ServerNode.Builder type`, it ends up hallucinating, as can be seen in the example generations with the text-davinci-003 and SantaCoder models. The completion uses identifiers `host` and `port`, which do not exist in the type `ServerNode.Builder`. The generated code therefore results in “symbol not found” compilation errors. 

MGD uses static analysis to guide the decoding of LMs, to generate code following certain properties. In the example, MGD is used to monitor for generating code with type-correct dereferences, and the SantaCoder model with the same prompt is able to generate the correct code completion, which compiles and matches the ground truth as well.

As reported in the paper, we observe that **MGD can improve the compilation rate of code generated by LMs at all scales (350M-175B) by 19-25%**, without any training/fine-tuning required. Further, it boosts the ground-truth match at all granularities from token-level to method-level code completion.

![](figures/motivating_example.png)

## 1. Datasets

### Dataset Statistics
|||
|--------------|:-----:|
| Number of repositories in PragmaticCode |  100 |
| Number of methods in DotPrompts | 1420 |
| Number of examples in DotPrompts | 10538 |

### PragmaticCode
PragmaticCode is a dataset of real-world open-source Java projects complete with their development environments and dependencies (through their respective build systems). The authors tried to ensure that all the repositories in PragmaticCode were released publicly only after the determined training dataset cutoff date (31 March 2022) for the CodeGen, SantaCoder and text-davinci-003 family of models, which were used to evaluate MGD.

The full dataset, along with repository zip files is available in our Zenodo dataset release at [https://zenodo.org/records/10072088](https://zenodo.org/records/10072088). The list of repositories along with their respective licenses consisting PragmaticCode is available in [datasets/PragmaticCode/repos.csv](datasets/PragmaticCode/repos.csv). The contents of the files required for inference for each of the repositories is available in [datasets/PragmaticCode/fileContentsByRepo.json](datasets/PragmaticCode/fileContentsByRepo.json).

### DotPrompts
DotPrompts is a set of examples derived from PragmaticCode, such that each example consists of a prompt to a dereference location (a code location having the "." operator in Java). DotPrompts can be used to benchmark Language Models of Code on their ability to utilize repository level context to generate code for method-level completion tasks. The task for the models is to complete a partially written Java method, utilizing the full repository available from PragmaticCode. Since all the repositories in PragmaticCode are buildable, DotPrompts (derived from PragmaticCode) supports Compilation Rate as a metric of evaluation for generated code, apart from standard metrics of ground truth match like Next-Identifier Match, Identifier Sequence Match and Prefix Match. 

The scenario described in [motivating example above](#monitor-guided-decoding-motivating-example) is an example in DotPrompts.

The complete description of an example in DotPrompts is a tuple - `(repo, classFileName, methodStartIdx, methodStopIdx, dot_idx)`. The dataset is available at [datasets/DotPrompts/dataset.csv](datasets/DotPrompts/dataset.csv).

## 2. Evaluation Scripts
### Environment Setup
We use the Python packages listed in [requirements.txt](requirements.txt). Our experiments used python 3.10. It is recommended to install the same with dependencies in an isolated virtual environment. To create a virtual environment using `venv`:
```setup
python3 -m venv venv_monitors4codegen
source venv_monitors4codegen/bin/activate
```
or using conda:
```
conda create -n monitors4codegen python=3.10
conda activate monitors4codegen
```
Further details and instructions on creation of python virtual environments can be found in the [official documentation](https://docs.python.org/3/library/venv.html). Further, we also refer users to [Miniconda](https://docs.conda.io/en/latest/miniconda.html), as an alternative to the above steps for creation of the virtual environment.

To install the requirements for running evaluations as described [below](#2-evaluation-scripts):

```setup
pip3 install -r requirements.txt
```

### Running the evaluation script
The evaluation script can be run as follows:
```
python3 eval_results.py <path to inference results - csv> <path to PragmaticCode filecontents - json> <path to output directory>
```

The above command will create a directory `<path to output directory>`, containing all the graphs and tables reported in the paper along with extra details. The command also generates a report in the output directory, named `Report.md` which relates the generated figures to sections in the paper. 

To ensure that the environment setup has been done correctly, please run the below command, which runs the evaluation script over dummy data (included in [inference_results/dotprompts_results_sample.csv](inference_results/dotprompts_results_sample.csv)). If the command fails, that indicates an error in the environment setup and the authors request you to kindly report the same.
```
python3 evaluation_scripts/eval_results.py inference_results/dotprompts_results_sample.csv datasets/PragmaticCode/fileContentsByRepo.json results_sample/
```

### Description of `inference results csv` file format
Description of expected columns in the inference results csv input to the evaluation script:
* `repo`: Name of the repository from which the testcase was sourced
* `classFileName`: relative path to file containing the testcase prompt location
* `methodStartIdx`: String index of starting `'{'` of the method
* `methodStopIdx`: String index of closing `'}'` of the method
* `dot_idx`: String index of `'.'` that is the dereference prompt point
* `configuration`: Identifies the configuration used to generate the given code sample. Values from: `['SC-classExprTypes', 'CG-6B', 'SC-FIM-classExprTypes', 'SC-RLPG-MGD', 'SC-MGD', 'SC-FIM-classExprTypes-MGD', 'CG-2B', 'SC', 'CG-2B-MGD', 'CG-350M-classExprTypes-MGD', 'SC-FIM', 'TD-3', 'CG-350M-MGD', 'SC-FIM-MGD', 'SC-RLPG', 'CG-350M', 'CG-350M-classExprTypes', 'SC-classExprTypes-MGD', 'CG-6B-MGD', 'TD-3-MGD']`
* `temperature`: Temperature used for sampling. Values from: `[0.8, 0.6, 0.4, 0.2]`
* `model`: Name of the model used for sampling. Values from: `['Salesforce/codegen-6B-multi', 'bigcode/santacoder', 'Salesforce/codegen-2B-multi', 'Salesforce/codegen-350M-multi', 'text-davinci-003']`
* `context`: Decoding strategy used. Values from: `['autoregressive', 'fim']`
* `prefix`: Prompt strategy used. Values from: `['classExprTypes', 'none', 'rlpg']`
* `rlpg_best_rule_name`: Name of the rule used for creating RLPG prompt (if used for the corresponding testcase). Values from: `[nan, 'in_file#lines#0.25', 'in_file#lines#0.5', 'in_file#lines#0.75', 'import_file#method_names#0.5']`
* `output`: Generated output by the model
* `compilationSucceeded`: Result of compiling the generated method in the context of the full repository. 1 if success, 0 otherwise. Values from: `[1, 0]`

## 3. Inference Results over DotPrompts
We provide all inferences (generated code) generated by all model configurations reported in the paper, for every example in DotPrompts. This consists of 6 independently sampled inferences for 18 different model configurations (spanning parameter scale, prompt templates, use of FIM context, etc.) for every example in DotPrompts.

The generated samples along with their compilation status, following the format described [above](#description-of-inference-results-csv-file-format), is available at [inference_results/dotprompts_results.csv](inference_results/dotprompts_results.csv). The file is stored using [git lfs](https://git-lfs.com/). If the file is not available locally after cloning this repository, please check the [git lfs website](https://git-lfs.com/) for instructions on setup, and clone the repository again after git lfs setup.

Each row in the file contains several multi-line string cells, and therefore, while viewing them in tools like Microsoft Office Excel, kindly enable "Word Wrap" to be able to view the full contents.

To run the [evaluation scripts](#2-evaluation-scripts) over the inferences, in order to reproduce the graphs and tables reported in the paper, run:
```
python3 evaluation_scripts/eval_results.py inference_results/dotprompts_results.csv datasets/PragmaticCode/fileContentsByRepo.json results/
```

The above command creates a directory [results](results/) (already included in the repository), containing all the figures and tables provided in the paper along with extra details. The command also generates a report in the output directory which relates the generated figures to sections in the paper. In case of above command, the report is generated at [results/Report.md](results/Report.md).

## 4. `multilspy`
**The `multilspy` library has now been migrated to [microsoft/multilspy](https://github.com/microsoft/multilspy).**

`multilspy` is a cross-platform library that we have built to set up and interact with various language servers in a unified and easy way. [Language servers]((https://microsoft.github.io/language-server-protocol/overviews/lsp/overview/)) are tools that perform a variety of static analyses on source code and provide useful information such as type-directed code completion suggestions, symbol definition locations, symbol references, etc., over the [Language Server Protocol (LSP)](https://microsoft.github.io/language-server-protocol/overviews/lsp/overview/). `multilspy` intends to ease the process of using language servers, by abstracting the setting up of the language servers, performing language-specific configuration and handling communication with the server over the json-rpc based protocol, while exposing a simple interface to the user.

Since LSP is language-agnostic, `multilspy` can provide the results for static analyses of code in different languages over a common interface. `multilspy` is easily extensible to any language that has a Language Server and currently supports Java, Rust, C# and Python and we aim to support more language servers from the [list of language server implementations](https://microsoft.github.io/language-server-protocol/implementors/servers/).

Some of the analyses results that `multilspy` can provide are:
- Finding the definition of a function or a class ([textDocument/definition](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_definition))
- Finding the callers of a function or the instantiations of a class ([textDocument/references](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_references))
- Providing type-based dereference completions ([textDocument/completion](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_completion))
- Getting information displayed when hovering over symbols, like method signature ([textDocument/hover](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_hover))
- Getting list/tree of all symbols defined in a given file, along with symbol type like class, method, etc. ([textDocument/documentSymbol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_documentSymbol))
- Please create an issue/PR to add any other LSP request not listed above

### Installation
To install `multilspy` using pip, execute the following command:
```
pip install https://github.com/microsoft/multilspy/archive/main.zip
```

### Usage
Example usage:
```python
from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
...
config = MultilspyConfig.from_dict({"code_language": "java"}) # Also supports "python", "rust", "csharp"
logger = MultilspyLogger()
lsp = SyncLanguageServer.create(config, logger, "/abs/path/to/project/root/")
with lsp.start_server():
    result = lsp.request_definition(
        "relative/path/to/code_file.java", # Filename of location where request is being made
        163, # line number of symbol for which request is being made
        4 # column number of symbol for which request is being made
    )
    result2 = lsp.request_completions(
        ...
    )
    result3 = lsp.request_references(
        ...
    )
    result4 = lsp.request_document_symbols(
        ...
    )
    result5 = lsp.request_hover(
        ...
    )
    ...
```

`multilspy` also provides an asyncio based API which can be used in async contexts. Example usage (asyncio):
```python
from multilspy import LanguageServer
...
lsp = LanguageServer.create(...)
async with lsp.start_server():
    result = await lsp.request_definition(
        ...
    )
    ...
```

The file [microsoft/multilspy - src/multilspy/language_server.py](https://github.com/microsoft/multilspy/blob/main/src/multilspy/language_server.py) provides the `multilspy` API. Several tests for `multilspy` present under [microsoft/multilspy - tests/multilspy/](https://github.com/microsoft/multilspy/tree/main/tests/multilspy) provide detailed usage examples for `multilspy`. The tests can be executed by running:
```bash
pytest tests/multilspy
```

## 5. Monitor-Guided Decoding

A monitor under the Monitor-Guided Decoding framework, is instantiated using `multilspy` as the LSP client, and provides maskgen to guide the LM decoding. The monitor interface is defined as class `Monitor` in file [src/monitors4codegen/monitor_guided_decoding/monitor.py](src/monitors4codegen/monitor_guided_decoding/monitor.py). The interface is implemented by various monitors supporting different properties like valid identifier dereferences, valid number of arguments, valid typestate method calls, etc.

### MGD with HuggingFace models
[src/monitors4codegen/monitor_guided_decoding/hf_gen.py](src/monitors4codegen/monitor_guided_decoding/hf_gen.py) provides the class `MGDLogitsProcessor` which can be used with any HuggingFace Language Model, as a [`LogitsProcessor`](https://huggingface.co/docs/transformers/internal/generation_utils#logitsprocessor) to guide the LM using MGD. Example uses with [SantaCoder](https://huggingface.co/bigcode/santacoder) model are available in [tests/monitor_guided_decoding/test_dereferences_monitor_java.py](tests/monitor_guided_decoding/test_dereferences_monitor_java.py).

### MGD with OpenAI models
[src/monitors4codegen/monitor_guided_decoding/openai_gen.py](src/monitors4codegen/monitor_guided_decoding/openai_gen.py) provides the method `openai_mgd` which takes the prompt and a `Monitor` as input, and returns the MGD guided generation using an OpenAI model.

### Monitors
#### Dereferences Monitor
[src/monitors4codegen/monitor_guided_decoding/monitors/dereferences_monitor.py](src/monitors4codegen/monitor_guided_decoding/monitors/dereferences_monitor.py) provides the instantiation of `Monitor` class for dereferences monitor. It can be used to guide LMs to generate valid identifier dereferences. Unit tests for the dereferences monitor are present in [tests/monitor_guided_decoding/test_dereferences_monitor_java.py](tests/monitor_guided_decoding/test_dereferences_monitor_java.py), which also provide usage examples for the dereferences monitor.

#### Monitor for valid number of arguments to function calls
[src/monitors4codegen/monitor_guided_decoding/monitors/numargs_monitor.py](src/monitors4codegen/monitor_guided_decoding/monitors/numargs_monitor.py) provides the instantiation of `Monitor` class for numargs_monitor. It can be used to guide LMs to generate correct number of arguments to function calls. Unit tests, which also provide usage examples are present in [tests/monitor_guided_decoding/test_numargs_monitor_java.py](tests/monitor_guided_decoding/test_numargs_monitor_java.py).

#### Monitor for typestate specifications
The typestate analysis is used to enforce that methods on an object are called in a certain order, consistent with the ordering constraints provided by the API contracts. Example usage of the typestate monitor for Rust is available in the unit test file [tests/monitor_guided_decoding/test_typestate_monitor_rust.py](tests/monitor_guided_decoding/test_typestate_monitor_rust.py).

#### Switch-Enum Monitor
[src/monitors4codegen/monitor_guided_decoding/monitors/switch_enum_monitor.py](src/monitors4codegen/monitor_guided_decoding/monitors/switch_enum_monitor.py) provides the instantiation of `Monitor` for generating valid named enum constants in C#. Unit tests for the switch-enum monitor are present in [tests/monitor_guided_decoding/test_switchenum_monitor_csharp.py](tests/monitor_guided_decoding/test_switchenum_monitor_csharp.py), which also provide usage examples for the switch-enum monitor.

#### Class Instantiation Monitor
[src/monitors4codegen/monitor_guided_decoding/monitors/class_instantiation_monitor.py](src/monitors4codegen/monitor_guided_decoding/monitors/class_instantiation_monitor.py) provides the instantiation of `Monitor` for generating valid class instantiations following `'new '` in a Java code base. Unit tests for the class-instantiation monitor, which provide examples usages are present in [tests/monitor_guided_decoding/test_classinstantiation_monitor_java.py](tests/monitor_guided_decoding/test_classinstantiation_monitor_java.py).

### Joint Monitoring
Multiple monitors can be used simultaneously to guide LMs to adhere to multiple properties. Example demonstration with 2 monitors used jointly are present in [tests/monitor_guided_decoding/test_joint_monitors.py](tests/monitor_guided_decoding/test_joint_monitors.py).

## Frequently Asked Questions (FAQ)
### ```asyncio``` related Runtime error when executing the tests for MGD
If you get the following error:
```
RuntimeError: Task <Task pending name='Task-2' coro=<_AsyncGeneratorContextManager.__aenter__() running at
    python3.8/contextlib.py:171> cb=[_chain_future.<locals>._call_set_state() at
    python3.8/asyncio/futures.py:367]> got Future <Future pending> attached to a different loop python3.8/asyncio/locks.py:309: RuntimeError
```

Please ensure that you create a new environment with Python ```>=3.10```. For further details, please have a look at the [StackOverflow Discussion](https://stackoverflow.com/questions/73599594/asyncio-works-in-python-3-10-but-not-in-python-3-8).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
