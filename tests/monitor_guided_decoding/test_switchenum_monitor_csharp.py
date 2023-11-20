"""
This file contains tests for Monitor-Guided Decoding for switch-enum in C#
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
from monitors4codegen.monitor_guided_decoding.monitors.switch_enum_monitor import SwitchEnumMonitor
from monitors4codegen.monitor_guided_decoding.monitors.dereferences_monitor import DereferencesMonitor
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.hf_gen import MGDLogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from monitors4codegen.multilspy.multilspy_types import Position
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import HFTokenizerWrapper

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_multilspy_csharp_ryujinx_switch_enum() -> None:
    """
    Test the working of SwitchEnumMonitor with C# repository - Ryujinx
    """
    code_language = Language.CSHARP
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/Ryujinx/Ryujinx/",
        "repo_commit": "e768a54f17b390c3ac10904c7909e3bef020edbd"
    }

    device = torch.device('cuda' if is_cuda_available() else 'cpu')

    model: transformers.modeling_utils.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "bigcode/santacoder", trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")

    with create_test_context(params) as context:
        lsp = SyncLanguageServer.create(context.config, context.logger, context.source_directory)
        with lsp.start_server():
            completions_filepath = "src/ARMeilleure/CodeGen/Arm64/CodeGenerator.cs"
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=1369, character=21),
                    Position(line=1385, character=8)
                )
                assert deleted_text == """AccessSize.Byte:
                    context.Assembler.Stlxrb(desired, address, result);
                    break;
                case AccessSize.Hword:
                    context.Assembler.Stlxrh(desired, address, result);
                    break;
                default:
                    context.Assembler.Stlxr(desired, address, result);
                    break;
            }

            context.Assembler.Cbnz(result, startOffset - context.StreamOffset); // Retry if store failed.

            context.JumpHere();

            context.Assembler.Clrex();
        """
                filebuffer = MonitorFileBuffer(
                    lsp.language_server,
                    completions_filepath,
                    (1369, 21),
                    (1369, 21),
                    code_language,   
                )
                monitor = SwitchEnumMonitor(HFTokenizerWrapper(tokenizer), filebuffer)
                mgd_logits_processor = MGDLogitsProcessor([monitor], lsp.language_server.server.loop)

                with open(str(PurePath(context.source_directory, completions_filepath)), "r") as f:
                    filecontent = f.read()

                pos_idx = TextUtils.get_index_from_line_col(filecontent, 1369, 21)
                assert filecontent[:pos_idx].endswith('case ')
                prompt = filecontent[:pos_idx]
                assert prompt[-1] == " "
                prompt_tokenized = tokenizer.encode(prompt, return_tensors="pt").cuda()[:, -(2048 - 512) :]

                generated_code_without_mgd = model.generate(
                    prompt_tokenized, do_sample=False, max_new_tokens=100, early_stopping=True
                )
                generated_code_without_mgd = tokenizer.decode(generated_code_without_mgd[0, -100:])

                assert (
                    generated_code_without_mgd
                    == "1:\n                    context.Assembler.Stb(result, Register(ZrRegister, result.Type));\n                    break;\n                case 2:\n                    context.Assembler.Stw(result, Register(ZrRegister, result.Type));\n                    break;\n                case 4:\n                    context.Assembler.Std(result, Register(ZrRegister, result.Type));\n                    break;\n                case 8:\n                    context.Assembler.Stq(result, Register(Zr"
                )

                # Generate code using santacoder model with the MGD logits processor and greedy decoding
                logits_processor = LogitsProcessorList([mgd_logits_processor])
                generated_code = model.generate(
                    prompt_tokenized,
                    do_sample=False,
                    max_new_tokens=100,
                    logits_processor=logits_processor,
                    early_stopping=True,
                )

                generated_code = tokenizer.decode(generated_code[0, -100:])

                assert (
                    generated_code
                    == "AccessSize.Byte:\n                    context.Assembler.Staxrb(actual, address);\n                    break;\n                case AccessSize.Hword:\n                    context.Assembler.Staxrh(actual, address);\n                    break;\n                default:\n                    context.Assembler.Staxr(actual, address);\n                    break;\n            }\n\n            context.Assembler.Cmp(actual, desired);\n\n            context.JumpToNear(ArmCondition.Eq);\n\n            context.Assembler.Staxr(result,"
                )
            


            completions_filepath = "src/ARMeilleure/CodeGen/X86/CodeGenerator.cs"
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=224, character=37),
                    Position(line=243, character=28)
                )
                assert deleted_text == """Intrinsic.X86Comisdlt:
                                    context.Assembler.Comisd(src1, src2);
                                    context.Assembler.Setcc(dest, X86Condition.Below);
                                    break;

                                case Intrinsic.X86Comisseq:
                                    context.Assembler.Comiss(src1, src2);
                                    context.Assembler.Setcc(dest, X86Condition.Equal);
                                    break;

                                case Intrinsic.X86Comissge:
                                    context.Assembler.Comiss(src1, src2);
                                    context.Assembler.Setcc(dest, X86Condition.AboveOrEqual);
                                    break;

                                case Intrinsic.X86Comisslt:
                                    context.Assembler.Comiss(src1, src2);
                                    context.Assembler.Setcc(dest, X86Condition.Below);
                                    break;
                            """
                filebuffer = MonitorFileBuffer(
                    lsp.language_server,
                    completions_filepath,
                    (224, 37),
                    (224, 37),
                    code_language,   
                )
                monitor = SwitchEnumMonitor(HFTokenizerWrapper(tokenizer), filebuffer)
                mgd_logits_processor = MGDLogitsProcessor([monitor], lsp.language_server.server.loop)

                with open(str(PurePath(context.source_directory, completions_filepath)), "r") as f:
                    filecontent = f.read()

                pos_idx = TextUtils.get_index_from_line_col(filecontent, 224, 37)
                assert filecontent[:pos_idx].endswith('case ')
                prompt = filecontent[:pos_idx]
                assert prompt[-1] == " "
                prompt_tokenized = tokenizer.encode(prompt, return_tensors="pt").cuda()[:, -(2048 - 512) :]

                generated_code_without_mgd = model.generate(
                    prompt_tokenized, do_sample=False, max_new_tokens=50, early_stopping=True
                )
                generated_code_without_mgd = tokenizer.decode(generated_code_without_mgd[0, -50:])

                assert (
                    generated_code_without_mgd
                    == " Intrinsic.X86Comisdgt:\n                                    context.Assembler.Comisd(src1, src2);\n                                    context.Assembler.Setcc(dest, X86Condition.GreaterThan);\n                                    break;\n\n                                case  In"
                )

                # Generate code using santacoder model with the MGD logits processor and greedy decoding
                logits_processor = LogitsProcessorList([mgd_logits_processor])
                generated_code = model.generate(
                    prompt_tokenized,
                    do_sample=False,
                    max_new_tokens=50,
                    logits_processor=logits_processor,
                    early_stopping=True,
                )

                generated_code = tokenizer.decode(generated_code[0, -50:])

                assert (
                    generated_code
                    == "Intrinsic.X86Comisdlt:\n                                    context.Assembler.Comisd(src1, src2);\n                                    context.Assembler.Setcc(dest, X86Condition.LessThan);\n                                    break;\n\n                                case Intrinsic"
                )

@pytest.mark.asyncio
@pytest.mark.skip(reason="TODO: This runs too slow. Reimplement joint monitoring")
async def test_multilspy_csharp_ryujinx_joint_switch_enum_dereferences() -> None:
    """
    Test the working of Joint monitoring with SwitchEnumMonitor and DereferencesMonitor with C# repository - Ryujinx
    """

    code_language = Language.CSHARP
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/Ryujinx/Ryujinx/",
        "repo_commit": "e768a54f17b390c3ac10904c7909e3bef020edbd"
    }

    device = torch.device('cuda' if is_cuda_available() else 'cpu')

    model: transformers.modeling_utils.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "bigcode/santacoder", trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")

    with create_test_context(params) as context:
        lsp1 = SyncLanguageServer.create(context.config, context.logger, context.source_directory)
        lsp2 = SyncLanguageServer.create(context.config, context.logger, context.source_directory)
        with lsp1.start_server(), lsp2.start_server():
            completions_filepath = "src/ARMeilleure/CodeGen/X86/CodeGenerator.cs"
            with lsp1.open_file(completions_filepath), lsp2.open_file(completions_filepath):
                deleted_text1 = lsp1.delete_text_between_positions(
                    completions_filepath,
                    Position(line=224, character=37),
                    Position(line=243, character=28)
                )
                deleted_text2 = lsp2.delete_text_between_positions(
                    completions_filepath,
                    Position(line=224, character=37),
                    Position(line=243, character=28)
                )
                assert deleted_text1 == deleted_text2
                assert deleted_text1 == """Intrinsic.X86Comisdlt:
                                    context.Assembler.Comisd(src1, src2);
                                    context.Assembler.Setcc(dest, X86Condition.Below);
                                    break;

                                case Intrinsic.X86Comisseq:
                                    context.Assembler.Comiss(src1, src2);
                                    context.Assembler.Setcc(dest, X86Condition.Equal);
                                    break;

                                case Intrinsic.X86Comissge:
                                    context.Assembler.Comiss(src1, src2);
                                    context.Assembler.Setcc(dest, X86Condition.AboveOrEqual);
                                    break;

                                case Intrinsic.X86Comisslt:
                                    context.Assembler.Comiss(src1, src2);
                                    context.Assembler.Setcc(dest, X86Condition.Below);
                                    break;
                            """
                filebuffer_enum = MonitorFileBuffer(
                    lsp1.language_server,
                    completions_filepath,
                    (224, 37),
                    (224, 37),
                    code_language,   
                )
                monitor_switch_enum = SwitchEnumMonitor(HFTokenizerWrapper(tokenizer), filebuffer_enum)
                mgd_logits_processor_switch_enum = MGDLogitsProcessor([monitor_switch_enum], lsp1.language_server.server.loop)
                
                filebuffer_dereferences = MonitorFileBuffer(
                    lsp2.language_server,
                    completions_filepath,
                    (224, 37),
                    (224, 37),
                    code_language,
                )
                monitor_dereferences = DereferencesMonitor(HFTokenizerWrapper(tokenizer), filebuffer_dereferences)
                mgd_logits_processor_dereferences = MGDLogitsProcessor([monitor_dereferences], lsp2.language_server.server.loop)

                with open(str(PurePath(context.source_directory, completions_filepath)), "r") as f:
                    filecontent = f.read()

                pos_idx = TextUtils.get_index_from_line_col(filecontent, 224, 37)
                assert filecontent[:pos_idx].endswith('case ')
                prompt = filecontent[:pos_idx]
                assert prompt[-1] == " "
                prompt_tokenized = tokenizer.encode(prompt, return_tensors="pt").cuda()[:, -(2048 - 512) :]

                generated_code_without_mgd = model.generate(
                    prompt_tokenized, do_sample=False, max_new_tokens=50, early_stopping=True
                )
                generated_code_without_mgd = tokenizer.decode(generated_code_without_mgd[0, -50:])

                assert (
                    generated_code_without_mgd
                    == " Intrinsic.X86Comisdgt:\n                                    context.Assembler.Comisd(src1, src2);\n                                    context.Assembler.Setcc(dest, X86Condition.GreaterThan);\n                                    break;\n\n                                case  In"
                )

                # Generate code using santacoder model with the MGD logits processor and greedy decoding
                logits_processor = LogitsProcessorList([mgd_logits_processor_switch_enum, mgd_logits_processor_dereferences])
                generated_code = model.generate(
                    prompt_tokenized,
                    do_sample=False,
                    max_new_tokens=50,
                    logits_processor=logits_processor,
                    early_stopping=True,
                )

                generated_code = tokenizer.decode(generated_code[0, -50:])

                assert (
                    generated_code
                    == "Intrinsic.X86Comisdlt:\n                                    context.Assembler.Comisd(src1, src2);\n                                    context.Assembler.Setcc(dest, X86Condition.Below);\n                                    break;\n\n                                case Intrinsic"
                )
