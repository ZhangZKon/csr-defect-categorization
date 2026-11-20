# Constrained Staged Refinement for Defect Categorization in Shared Multi-Task Learning

Research code for the paper *Constrained Staged Refinement for Defect Categorization in Shared Multi-Task Learning*.

The repository implements a staged refinement pipeline for defect categorization with shared multi-task representations, together with ablation and cross-backbone experiments.

## Repository structure

```text
Models/
  CSR/
    CompDefect/       # reproduced COMPDEFECT-style baseline
    Stage1/            # first refinement stage
    Stage2/            # edit-signal refinement
    Stage3/            # low-learning-rate consolidation
  ablation_study/     # controlled ablations
  cross-backbone/     # backbone comparison experiments
parser/               # code parsing and data-flow utilities
Data/                 # label metadata and local data location
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default encoder is `microsoft/graphcodebert-base`.

## Data

The original CSV files are intentionally excluded from this repository. They contain source-code snippets from third-party projects and should be obtained and redistributed only under the applicable dataset terms.

Place authorized files under `Data/`:

```text
Data/
  train_data.csv
  val_data.csv
  test_data.csv
  type2label_all.json
```

Related upstream implementation:

- [CompDefect](https://github.com/jacknichao/CompDefect)

## Running the staged pipeline

Run commands from the repository root.

```bash
cd Models/CSR/CompDefect
bash train_all.sh
bash eval_all.sh
```

Then run the refinement stages:

```bash
cd ../Stage1
bash train_all.sh
bash eval_all.sh

cd ../Stage2
python run.py \
  --output_dir saved_models \
  --config config_all.yml \
  --load_model_path ../Stage1/saved_models/checkpoint-best-f1/pytorch_model.bin

cd ../Stage3
python run.py \
  --output_dir saved_models \
  --config config_all.yml \
  --load_model_path ../Stage2/saved_models/checkpoint-best-f1/pytorch_model.bin
```

Ablation and cross-backbone variants are available under `Models/ablation_study/` and `Models/cross-backbone/`.
