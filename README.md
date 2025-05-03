ReQAP
============
**Re**cursive **Q**uestion Understanding for Complex Question **A**nswering over Heterogeneous **P**ersonal Data
---------------

- [Description](#description)
- [Code](#code)
  - [System requirements](#system-requirements)
  - [Installation](#installation)
  - [ReQAP - Inference](#reqap---inference)
  - [ReQAP - Full training procedure](#reqap---full-training-procedure)
  - [Baselines](#baselines)
    - [RAG - Inference](#rag---inference)
    - [RAG - Full training procedure](#rag---full-training-procedure)
    - [Query Generation - Inference](#query-generation---inference)
    - [Query Generation - Full training procedure](#query-generation---full-training-procedure)
- [Feedback](#feedback)
- [License](#license)
- [Acknowledgements](#acknowledgements)


# Description
This repository contains the code and data for our ReQAP paper on "Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data".

*Question answering over mixed sources, like text and tables, has been advanced by verbalizing all contents and encoding it with a language model. A prominent case of such heterogeneous data is personal information: user devices log vast amounts of data every day, such as calendar entries, workout statistics, shopping records, streaming history, and more. Information needs range from simple look-ups to queries of analytical nature. The challenge is to provide humans with convenient access with small footprint, so that all personal data stays on the user devices. We present ReQAP, a novel method that creates an executable operator tree for a given question, via recursive decomposition. Operators are designed to enable seamless integration of structured and unstructured sources, and the execution of the operator tree yields a traceable answer. We further release the PerQA benchmark, with persona-based data and questions, covering a diverse spectrum of realistic user needs.*


If you use this code, please cite:
```bibtex
@article{christmann2025recursive,
  title={Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data},
  author={Christmann, Philipp and Weikum, Gerhard},
  journal={arXiv},
  year={2025}
}
```

# Code

## System requirements

All code was tested on Linux only.
- Conda
- PyTorch
- GPU (for training)

## Installation
We recommend the installation via conda, and provide the corresponding environment file in [conda-reqap.yml](conda-reqap.yml):

```bash
    git clone https://github.com/PhilippChr/ReQAP.git
    cd ReQAP/
    conda env create --file conda-reqap.yml
    conda activate reqap
    pip install -e .
```

Alternatively, you can also install the requirements via pip, using the [requirements.txt](requirements.txt) or [requirements-cpu.txt](requirements-cpu.txt) file. In this case, for running the code via GPU, further packages might be required.

To initialize the repo (download data, benchmark, models), run:
```bash
bash scripts/initialize.sh
```

## ReQAP - Inference

* ReQAP SFT
    - Run QUD stage
        ```bash
        bash scripts/pipeline.sh --qud-test config/perqa/reqap_sft.yml  # much faster with GPU
        ```
    - Run OTX stage
        ```bash
        bash scripts/pipeline.sh --otx-test config/perqa/reqap_sft.yml  # much faster with GPU
        ```

* ReQAP with LLaMA
    - Run QUD stage
        ```bash
        bash scripts/pipeline.sh --qud-test config/perqa/reqap_llama.yml  # much faster with GPU
        ```
    - Run OTX stage
        ```bash
        bash scripts/pipeline.sh --otx-test config/perqa/reqap_llama.yml  # much faster with GPU
        ```

* ReQAP with GPT
    - Add your OpenAI credentials in `config/perqa/reqap_openai.yml`
    - Run QUD stage
        ```bash
        bash scripts/pipeline.sh --qud-test config/perqa/reqap_openai.yml
        ```
    - Run OTX stage
        ```bash
        bash scripts/pipeline.sh --otx-test config/perqa/reqap_openai.yml  # much faster with GPU
        ```


## ReQAP - Full training procedure

This requires adding your OpenAI credentials in `config/perqa/reqap_openai.yml`,
or replacing `config/perqa/reqap_openai.yml` with `config/perqa/reqap_llama.yml`.

1. Run QUD stage via ICL on all train + dev questions 
    - Run QUD-ICL on train set
        ```bash
        bash scripts/pipeline.sh --create_qu_plans-train config/perqa/reqap_openai.yml  # requires GPU/API
        ```
    - Run QUD-ICL on dev set
        ```bash
        bash scripts/pipeline.sh --create_qu_plans-dev config/perqa/reqap_openai.yml  # requires GPU/API
        ```

2. Train RETRIEVE and EXTRACT operators
    * RETRIEVE
        - Detect all RETRIEVE calls
            ```bash
            bash scripts/run_retrieval.sh --derive_retrieve_calls config/perqa/reqap_openai.yml
            ```
        - Construct SPLADE indices
            ```bash
            bash scripts/run_retrieval.sh --construct_index config/perqa/reqap_openai.yml  # much faster with GPU
            ```
        - Prepare data for training RETRIEVE models
            ```bash
            bash scripts/prepare_retrieval_data.sh config/perqa/reqap_openai.yml  # CPU: runs 14 parallel scripts
            ```
        - Merge RETRIEVE training data for all personas
            ```bash
            bash scripts/merge_retrieval_data.sh data/training_data/perqa
            ```
        - Train RETRIEVE models of size L (default)
            ```bash
            bash scripts/run_retrieval.sh --train_ce_events config/perqa/training/reqap_ce_events-ms-marco-MiniLM-L-12.yml  # requires GPU
            bash scripts/run_retrieval.sh --train_ce_patterns config/perqa/training/reqap_ce_patterns-ms-marco-MiniLM-L-12.yml  # requires GPU
            ```
        - [OPTIONAL] Train RETRIEVE models of size M
            ```bash
            bash scripts/run_retrieval.sh --train_ce_events config/perqa/training/reqap_ce_events-ms-marco-MiniLM-L-6.yml  # requires GPU
            bash scripts/run_retrieval.sh --train_ce_patterns config/perqa/training/reqap_ce_patterns-ms-marco-MiniLM-L-6.yml  # requires GPU
            ```
        - [OPTIONAL] Train RETRIEVE models of size S
            ```bash
            bash scripts/run_retrieval.sh --train_ce_events config/perqa/training/reqap_ce_events-ms-marco-MiniLM-L-2.yml  # requires GPU
            bash scripts/run_retrieval.sh --train_ce_patterns config/perqa/training/reqap_ce_patterns-ms-marco-MiniLM-L-2.yml  # requires GPU
            ```
        - [OPTIONAL] Train RETRIEVE models of size XS
            ```bash
            bash scripts/run_retrieval.sh --train_ce_events config/perqa/training/reqap_ce_events-ms-marco-TinyBERT-L-2.yml  # requires GPU
            bash scripts/run_retrieval.sh --train_ce_patterns config/perqa/training/reqap_ce_patterns-ms-marco-TinyBERT-L-2.yml  # requires GPU
            ```
        
    * EXTRACT
        - Derive EXTRACT calls with related attributes 
            ```bash
            bash scripts/run_extract.sh --derive_attributes config/perqa/reqap_openai.yml
            ```
        - Identify aliases for keys in EXTRACT calls
            ```bash
            bash scripts/run_extract.sh --derive_attribute_mappings config/perqa/reqap_openai.yml  # requires GPU/API
            ```
        - Derive training data for EXTRACT model
            ```bash
            bash scripts/run_extract.sh --derive_data config/perqa/reqap_openai.yml
            ```
        - Train EXTRACT model of size L (default)
            ```bash
            bash scripts/run_extract.sh --train config/perqa/training/reqap_extract-bart-base.yml  # requires GPU
            ```
        - [OPTIONAL] Train EXTRACT model of size M
            ```bash
            bash scripts/run_extract.sh --train config/perqa/training/reqap_extract-bart-small.yml  # requires GPU
            ```
        - [OPTIONAL] Train EXTRACT model of size S
            ```bash
            bash scripts/run_extract.sh --train config/perqa/training/reqap_extract-t5-efficient-mini.yml  # requires GPU
            ```
        - [OPTIONAL] Train EXTRACT model of size XS
            ```bash
            bash scripts/run_extract.sh --train config/perqa/training/reqap_extract-t5-efficient-tiny.yml  # requires GPU
            ```
    
3. Derive data for model distillation 
    - [OPTION 1] Identify correct operator trees (in single runs)
        ```bash
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml  # requires GPU
        bash scripts/pipeline.sh --loop-dev config/perqa/reqap_openai.yml  # requires GPU
        ```
    - [OPTION 2] Identify correct operator trees (run individually per persona)
        ```bash
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_0  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_1  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_2  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_3  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_4  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_5  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_6  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_7  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_8  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_9  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_10  # requires GPU
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_11  # requires GPU
        bash scripts/pipeline.sh --loop-dev config/perqa/reqap_openai.yml dev_persona_0  # requires GPU
        bash scripts/pipeline.sh --loop-dev config/perqa/reqap_openai.yml dev_persona_1  # requires GPU
        ```
    - [OPTION 3] Identify correct operator trees for single train persona (much faster, less training data)
        ```bash
        bash scripts/pipeline.sh --loop-train config/perqa/reqap_openai.yml train_persona_0  # requires GPU
        bash scripts/pipeline.sh --loop-dev config/perqa/reqap_openai.yml dev_persona_0  # requires GPU
        bash scripts/pipeline.sh --loop-dev config/perqa/reqap_openai.yml dev_persona_1  # requires GPU
        ```

4. Merge data for model distillation
    ```bash
    bash scripts/merge_qu_data.sh 
    ```

5. Train QUD stage
    - Derive training data for QUD stage
        ```bash
        bash scripts/run_qu.sh --derive_data config/perqa/reqap_sft.yml
    - Train QUD model of size M (default)
        ```bash
        bash scripts/run_qu.sh --train config/perqa/training/reqap_qu-causal-llama1b.yml  # requires GPU
        ```
    - [OPTIONAL] Train QUD model of size L
        ```bash
        bash scripts/run_qu.sh --train config/perqa/training/reqap_qu-causal-llama3b.yml  # requires GPU
        ```
    - [OPTIONAL] Train QUD model of size S
        ```bash
        bash scripts/run_qu.sh --train config/perqa/training/reqap_qu-causal-hf-smollm2-360M.yml  # requires GPU
        ```
    - [OPTIONAL] Train QUD model of size XS
        ```bash
        bash scripts/run_qu.sh --train config/perqa/training/reqap_qu-causal-hf-smollm2-135M.yml  # requires GPU
        ```

6. Run ReQAP (SFT) inference
    - Run QUD stage
        ```bash
        bash scripts/pipeline.sh --qud-test config/perqa/reqap_sft.yml  # much faster with GPU
        ```
    - Run OTX stage
        ```bash
        bash scripts/pipeline.sh --otx-test config/perqa/reqap_sft.yml  # much faster with GPU
        ```

## Baselines

### RAG - Inference

* RAG SFT (requires following training below)
    - Run retrieval
        ```bash
        bash scripts/rag.sh --retrieve config/perqa/rag_openai.yml test  # much faster with GPU
        ```
    - Run generation
        ```bash
        bash scripts/rag.sh --test config/perqa/rag_sft.yml  # requires GPU
        ```

* RAG with LLaMA
    - Run retrieval
        ```bash
        bash scripts/rag.sh --retrieve config/perqa/rag_openai.yml test  # much faster with GPU
        ```
    - Run generation
        ```bash
        bash scripts/rag.sh --test config/perqa/rag_llama.yml  # requires GPU
        ```

* RAG with GPT
    - Add your OpenAI credentials in `config/perqa/rag_openai.yml`
    - Run retrieval
        ```bash
        bash scripts/rag.sh --retrieve config/perqa/rag_openai.yml test  # much faster with GPU
        ```
    - Run generation
        ```bash
        bash scripts/rag.sh --test config/perqa/rag_openai.yml  # requires GPU
        ```


### RAG - Full training procedure

1. Train retrieval for RAG baseline 
    - Derive training data (makes use of the ReQAP retrieval training data; assumed to be there already)
        ```bash
        bash scripts/rag.sh --ce_derive_data config/perqa/rag_openai.yml
        ```
    - Train the cross-encoder
        ```bash
        bash scripts/rag.sh --ce_train config/perqa/rag_openai.yml  # requires GPU
        ```

2. Run retrieval inference
    ```bash
    bash scripts/rag.sh --retrieve config/perqa/rag_openai.yml train  # much faster with GPU
    bash scripts/rag.sh --retrieve config/perqa/rag_openai.yml dev  # much faster with GPU
    bash scripts/rag.sh --retrieve config/perqa/rag_openai.yml test  # much faster with GPU
    ```

3. Train answering model
    - Derive training data
        ```bash
        bash scripts/rag.sh --derive_data config/perqa/rag_sft.yml
        ```
    - Train model
        ```bash
        bash scripts/rag.sh --train config/perqa/rag_sft.yml  # requires GPU
        ```

4. Inference
    ```bash
    bash scripts/rag.sh --test config/perqa/rag_sft.yml  # requires GPU
    ```


### Query Generation - Inference

* CodeGen SFT (requires following training below)
    ```bash
    bash scripts/query_generation.sh --test config/perqa/query_generation_sft.yml # requires GPU
    ```

* CodeGen with LLaMA
    ```bash
    bash scripts/query_generation.sh --test config/perqa/query_generation_llama.yml # requires GPU
    ```

* CodeGen with GPT
    - Add your OpenAI credentials in `config/perqa/query_generation_openai.yml`
    ```bash
    bash scripts/query_generation.sh --test config/perqa/query_generation_openai.yml # requires GPU
    ```

### Query Generation - Full training procedure

1. Prepare training data
    ```bash
    bash scripts/query_generation.sh --derive_data config/perqa/query_generation_sft.yml
    ```
2. Train translation model
    ```bash
    bash scripts/query_generation.sh --train config/perqa/query_generation_sft.yml # requires GPU
    ```
3. Inference
    ```bash
    bash scripts/query_generation.sh --test config/perqa/query_generation_sft.yml # requires GPU
    ```


# Feedback
We tried our best to document the code of this project, and make it accessible for easy usage.
If you feel that some parts of the documentation/code could be improved, or have other feedback,
please do not hesitate and let us know!

You can contact us via mail: [pchristm@mpi-inf.mpg.de](mailto:pchristm@mpi-inf.mpg.de).
Any feedback (also positive ;) ) is much appreciated!

# License
The ReQAP project by [Philipp Christmann](https://people.mpi-inf.mpg.de/~pchristm/) and [Gerhard Weikum](https://people.mpi-inf.mpg.de/~weikum/) is licensed under a [MIT license](LICENSE).

# Acknowledgements
Our retrieval utilizes SPLADE (https://github.com/naver/splade).
We adapt parts of their code in this repository.
