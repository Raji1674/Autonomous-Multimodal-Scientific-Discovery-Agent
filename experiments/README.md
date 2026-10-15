# Member 3 — ML Experiments, Dataset Discovery & Critic

## Overview

This module is responsible for discovering suitable datasets, validating data quality, running machine learning experiments, evaluating model performance, and reviewing experiment results.

## Responsibilities

- **Dataset Discovery:** Search public dataset repositories using research hypotheses and relevant keywords.
- **Dataset Loading:** Load datasets for experimentation.
- **Dataset Validation:** Check dataset structure, missing values, duplicate records, target columns, and potential data quality issues.
- **Experiment Training:** Train suitable baseline machine learning models.
- **Metrics Evaluation:** Calculate appropriate performance metrics.
- **Critic Module:** Review experiment outputs, identify potential issues, and report limitations.

## Current Implementation

- Sample Breast Cancer dataset used to test the initial ML pipeline.
- Dataset validation implemented and tested on the sample dataset.
- Baseline Logistic Regression training implemented.
- Evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrix.
- Critic module performs basic consistency checks on experiment results.
- Dataset discovery module is being developed to search OpenML and Hugging Face datasets.

## Experiment Workflow

1. Receive a research hypothesis.
2. Search for relevant datasets.
3. Inspect dataset metadata and source information.
4. Validate the dataset before training.
5. Select an appropriate experiment and evaluation strategy.
6. Train and evaluate the model.
7. Critique the results and report limitations.

## Output Files

Generated experiment outputs are stored locally in the `outputs/` directory, including:

- `experiment_results.json`
- `metrics.json`
- `critic_report.json`

Generated outputs and temporary files are excluded from Git where appropriate.

## Current Limitations

- Dataset discovery results are candidates, not automatically approved datasets.
- Dataset suitability, licensing, and target availability require validation.
- Autonomous experiment selection and integration with the other agents are still in progress.
- Initial model performance on a sample dataset does not establish scientific validity.

## Future Work

- Integrate paper-derived dataset references from the literature/RAG module.
- Improve dataset relevance scoring and validation.
- Support classification and regression workflows.
- Add experiment comparison and statistical significance analysis.
- Integrate with the code execution sandbox and main agent pipeline.
- Produce structured results for the dashboard and synthesis agent.