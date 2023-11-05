# Responsible AI Transparency Information
## What is Monitor-Guided Decoding (MGD)?  
Monitor-Guided Decoding (MGD) is a tool for Language Models (LMs) to generate more reliable code. It combines the token-by-token LM decoding with Program Analysis techniques (a method that can check the syntax, semantics, and logic of code, such as the ones used in Integrated Development Environments). Under MGD, a software called monitor runs concurrently to the decoder, and iteratively uses results from continuous program analysis to prevent generation of potentially problematic tokens, such as identifiers that are inconsistent with the type definitions. For example, a type analysis is performed at identifier dereferences, to find the list of type-correct symbols, and prevent generation of type-invalid symbols, thus generating code free from a large class of compilation errors. 

The static analysis in MGD is powered by Language Servers served over the Language Server Protocol. MGD takes as input a code repository, a partially completed code file within the repository, a prompt for the LM to generate the remaining code, and then uses a Language Model (from HuggingFace or OpenAI), to provide a code completion for it, while adhering to the monitored property. 

## What can Monitor-Guided Decoding do?   

MGD can improve the quality and reliability of code generation by LMs, especially when the code involves using types or functionality defined in another module, library, or when the LM has not seen such types or functionality during training (for example, the library version has upgraded with new APIs defined or private codebases). MGD can also prevent the LM from hallucinating non-existent dereferenced identifiers. Since MGD is prompt-agnostic, it can be used for various code generation tasks, such as code writing, code repair, code refactoring, code completion, etc., simply by changing the prompt. MGD can also be applied to any programming language for which a Language Server (The Language Server must declare “textDocument/completion” capability) is available. 

## What is/are Monitor-Guided Decoding’s intended use(s)?  

MGD is intended to be used as a research tool to advance the state of the art in and explore the potential of combining LM decoding with Program Analysis for code generation. It is also intended to be used as a baseline for evaluating and improving the performance of LMs on code generation tasks. It can be integrated in IDEs with LM based code-completion assistants; however, this use case has not been evaluated with users. MGD is not intended to be used as a substitute for human verification or testing of the generated code and does not provide guarantees for the generated code to be bug-free. 

## How was Monitor-Guided Decoding evaluated? What metrics are used to measure performance?  

MGD was evaluated on a dataset of open-source Java repositories from GitHub, called PragmaticCode, which contains code snippets with different levels of complexity and context. The dataset was used to curate a code benchmark, called DotPrompts (consisting of >10,000 testcases), which consists of prompts that require the LM to generate the remaining code for a partially completed nontrivial method. The benchmark is set up such that the LM must generate non-local identifier dereferences to complete the method.  

MGD was applied to several off-the-shelf LMs of different sizes and domains, such as CodeGen-{350M, 2B, 6B}-Multi, SantaCoder-1.1B, and OpenAI text-davinci-003. The performance of LMs with and without MGD was measured using the following metrics: 

1. Compilation Rate: Fraction of test cases, for which generated code compiled successfully 
2. Next Identifier Match: Fraction of test cases, for which generated next identifier is accurate 
3. Identifier Sequence Match: Percent prefix of ordered identifiers in the ground truth matched by the generated code 
4. Prefix Match: Percent prefix of ground truth matched by generated code 

The metrics were aggregated over 6 indepedent trials for each testcase using the following aggregation: 
* score@k - estimate of best score achievable by the evaluated model, given k independent trials. 

The results show that MGD consistently improved the ability of the LMs to generate code that compiles and matches the ground truth, across different metrics and models. MGD also outperformed the prompting technique on most metrics. MGD also demonstrated that LMs with fewer parameters, when guided with MGD, can outperform larger LMs without MGD. 

## What are the limitations of Monitor-Guided Decoding? How can users minimize the impact of Monitor-Guided Decoding’s limitations when using the system?  

MGD has some limitations that users should be aware of when using the system. Some of these limitations are: 
* The current instantiation of MGD monitors for type-consistent use of identifiers, which is one of the major sources of compilation errors in LM based code generation. However, there are other types of errors or bugs that MGD does not monitor or prevent, such as logical, syntactic, semantic, or runtime errors. Users should not rely on MGD to generate error-free code and should always verify and test the generated code for correctness and functionality. 
* MGD relies on the availability and accuracy of a Language Server for the programming language of interest. If the Language Server is not available, not compatible, or not reliable, MGD cannot be applied or may produce incorrect results. Users should ensure that the Language Server used is suitable and trustworthy.  
* MGD introduces some latency overhead to the code generation process, as it requires invoking the language server and masking the LM output iteratively. In our experiments, we find the latency overhead to not be significant, however, it may vary depending on the complexity of the code repository, size of the LM, speed of the static analysis, and the hardware and software configuration of the system. 
* MGD is a research tool that has not been extensively tested or validated with human users. It may not generalize well to domains and tasks that are beyond the scope of evaluation. 

## What operational factors and settings allow for effective and responsible use of Monitor-Guided Decoding?  

MGD has been shown to enhance the output of the LM by preventing a class of errors appearing in the generated code. However, the underlying generated code is still limited by the capability of the base LM.  
Some of the operational factors and settings that can enable effective and responsible use of MGD are: 
* Choosing an appropriate LM for the code generation task and the programming language of interest. Users should select an LM that has been trained on a relevant and diverse corpus of code. Users should also be aware of the limitations and assumptions of the LM, and how they may affect the code generation quality and reliability.  
* Reviewing and testing the generated code for correctness and functionality. Users should not blindly trust or use the generated code without verifying and testing it for errors, bugs, or vulnerabilities. Users should also document and acknowledge the use of MGD and the LM for their code generation task and cite the relevant sources and references.
