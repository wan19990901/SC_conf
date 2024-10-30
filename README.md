# RASC: Reasoning-Aware Self-Consistency for LLM Reasoning

This repository contains the implementation of the Reasoning-Aware Self-Consistency (RASC) framework, a novel approach that enhances sampling efficiency and reasoning faithfulness in Large Language Models (LLMs) by dynamically evaluating both outputs and rationales.

## Overview

RASC is designed to improve the efficiency and faithfulness of LLM reasoning by:
- Dynamically evaluating both reasoning paths and final answers
- Optimizing sampling efficiency while maintaining accuracy
- Enabling more informed sampling decisions and rationale selection
- Reducing sample usage by 60-80% while maintaining accuracy compared to existing methods

## Repository Structure

```
├── src/
│   ├── experiment_collection/    # Contains experimental results presented in the paper
│   ├── CS_based_early_stopping.py   # Implementation of early stopping mechanism
│   ├── CS_feature_extractor.py      # Feature extraction for reasoning evaluation
│   ├── IDV_CS_Model.py              # Core RASC model implementation
│   ├── LLM_agent.py                 # Interface for LLM interactions
│   ├── Parsers.py                   # Parsing utilities for model outputs
│   ├── SC_generator.py              # Self-consistency sample generation
│   ├── data_cleaning.py             # Data preprocessing utilities
│   ├── experiment.sh                # Experiment execution script
│   ├── human_eval.md                # Human evaluation guidelines
│   └── utils.py                     # Utility functions
├── data/
│   ├── question_data/               # Original question samples
│   │   └── preprocessed/            # Preprocessed question data
│   ├── other_data.txt              # Additional data files
│   └── result/                      # Experimental results
└── requirements.txt                 # Project dependencies
```

## Features

- Implementation of the RASC framework for efficient LLM reasoning
- Support for multiple LLM models (LLAMA2-7B, GPT3.5-turbo/GPT4, Vicuna-13B)
- Comprehensive feature extraction for reasoning quality assessment
- Early stopping mechanisms for optimized sampling
- Evaluation tools including human evaluation protocols

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Self-Consistency Chains (for extracting the answer):
```bash
python src/SC_generator.py
```

2. Data Preprocessing (for extracting the answer):
```bash
python src/data_cleaning.py 
```

3. Feature Extraction:
```bash
python src/CS_feature_extractor.py
```

4. Run Experiments:
```bash
bash src/experiment.sh
```

## Experiment Results

The `experiment_collection` directory contains all experimental results presented in the paper, including:
- Performance comparisons across different reasoning tasks
- Efficiency metrics and sample usage statistics
- Model comparisons and ablation studies

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{RASC2024,
  title={RASC: Reasoning-Aware Self-Consistency for Efficient and Faithful LLM Reasoning},
  author={Anonymous},
  journal={ACL submission},
  year={2024}
}
```

## Contact

For questions and feedback, please open an issue in this repository.

## Acknowledgments

We thank all contributors and reviewers who helped improve this work.