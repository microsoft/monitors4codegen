"""
This file contains tests for Monitor-Guided Decoding for valid class instantiations in Java
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
from monitors4codegen.monitor_guided_decoding.monitors.class_instantiation_monitor import ClassInstantiationMonitor
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.hf_gen import MGDLogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from monitors4codegen.multilspy.multilspy_types import Position
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import HFTokenizerWrapper

pytest_plugins = ("pytest_asyncio",)

@pytest.mark.asyncio
async def test_multilspy_java_example_repo_class_instantiation() -> None:
    """
    Test the working of ClassInstantiationMonitor with Java repository - ExampleRepo
    """
    code_language = Language.JAVA
    params = {
        "code_language": code_language,
        "repo_url": "https://github.com/LakshyAAAgrawal/ExampleRepo/",
        "repo_commit": "f3762fd55a457ff9c6b0bf3b266de2b203a766ab",
    }

    device = torch.device('cuda' if is_cuda_available() else 'cpu')

    model: transformers.modeling_utils.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "bigcode/santacoder", trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")

    with create_test_context(params) as context:
        lsp = SyncLanguageServer.create(context.config, context.logger, context.source_directory)
        with lsp.start_server():
            completions_filepath = "Main.java"
            with lsp.open_file(completions_filepath):
                deleted_text = lsp.delete_text_between_positions(
                    completions_filepath,
                    Position(line=16, character=24),
                    Position(line=36, character=5)
                )
                assert deleted_text == """Student("Alice", 10);
        Person p2 = new Teacher("Bob", "Science");
        
        // Create some course objects
        Course c1 = new Course("Math 101", t1, mathStudents);
        Course c2 = new Course("English 101", t2, englishStudents);
        
        // Print some information about the objects
        
        System.out.println("Person p1's name is " + p1.getName());
        
        System.out.println("Student s1's name is " + s1.getName());
        System.out.println("Student s1's id is " + s1.getId());
        
        System.out.println("Teacher t1's name is " + t1.getName());
        System.out.println("Teacher t1's subject is " + t1.getSubject());
        
        System.out.println("Course c1's name is " + c1.getName());
        System.out.println("Course c1's teacher is " + c1.getTeacher().getName());
        
     """
                prompt_pos = (16, 24)

                with open(str(PurePath(context.source_directory, completions_filepath)), "r") as f:
                    filecontent = f.read()

                pos_idx = TextUtils.get_index_from_line_col(filecontent, prompt_pos[0], prompt_pos[1])
                assert filecontent[:pos_idx].endswith('new ')

                prompt = filecontent[:pos_idx]
                assert filecontent[pos_idx-1] == " "
                prompt_tokenized = tokenizer.encode(prompt, return_tensors="pt").cuda()[:, -(2048 - 512) :]

                generated_code_without_mgd = model.generate(
                    prompt_tokenized, do_sample=False, max_new_tokens=30, early_stopping=True
                )
                generated_code_without_mgd = tokenizer.decode(generated_code_without_mgd[0, -30:])

                assert (
                    generated_code_without_mgd
                    == " Person(\"John\", \"Doe\", \"123-4567\", \"kenaa@example.com\", \"1234"
                )

                filebuffer = MonitorFileBuffer(
                    lsp.language_server,
                    completions_filepath,
                    prompt_pos,
                    prompt_pos,
                    code_language,   
                )
                monitor = ClassInstantiationMonitor(HFTokenizerWrapper(tokenizer), filebuffer)
                mgd_logits_processor = MGDLogitsProcessor([monitor], lsp.language_server.server.loop)

                # Generate code using santacoder model with the MGD logits processor and greedy decoding
                logits_processor = LogitsProcessorList([mgd_logits_processor])
                generated_code = model.generate(
                    prompt_tokenized,
                    do_sample=False,
                    max_new_tokens=30,
                    logits_processor=logits_processor,
                    early_stopping=True,
                )

                generated_code = tokenizer.decode(generated_code[0, -30:])

                assert (
                    generated_code
                    == "Student(\"John\", 1001);\n        Person p2 = new Student(\"Mary\", 1002);\n        Person p"
                )
