# Fine-tune with TEA

This repository provides an example script (tea_ft.py) to fine-tune BioBERT base cased v1.2 models with various TEA datasets by using Hugging Face machine learning libraries. An example result set from running the script for Pathogen Identifier and Strain Tagger corpora is provided below (median results for augmentation experiment/six random seeds/two epochs). The performance is measured by two different test datasets. Regular test dataset contains the baseline examples, while the enriched dataset has similar examples with approximately seven times more unique species and strain names, and can therefore be considered to measure potential overfitting to the species and strain names.

### Pathogen identifier dataset

<table>
<tr><th>Baseline test dataset</th><th>Enriched test dataset</th></tr>
<tr><td>

| Training dataset | Loss    | Precision | Recall | F1     |
|------------------|---------|-----------|--------|--------|
| Augmented        | 0.01    | 0.516     | 0.474  | 0.491  |
| Unaugmented      | 0.007   | 0.566     | 0.61   | 0.586  |

</td><td>

| Training dataset | Loss    | Precision | Recall | F1     |
|------------------|---------|-----------|--------|--------|
| Augmented        | 0.015   | 0.607     | 0.588  | 0.597  |
| Unaugmented      | 0.022   | 0.472     | 0.18   | 0.262  |

</td></tr></table>

### Strain tagger dataset

<table>
<tr><th>Baseline test dataset</th><th>Enriched test dataset</th></tr>
<tr><td>

| Training dataset | Loss    | Precision | Recall | F1     |
|------------------|---------|-----------|--------|--------|
| Augmented        | 0.032   | 0.567     | 0.534  | 0.551  |
| Unaugmented      | 0.026   | 0.553     | 0.56   | 0.561  |

</td><td>

| Training dataset | Loss    | Precision | Recall | F1     |
|------------------|---------|-----------|--------|--------|
| Augmented        | 0.036   | 0.665     | 0.748  | 0.706  |
| Unaugmented      | 0.049   | 0.632     | 0.529  | 0.572  |

</td></tr></table>

**Please note**: it is very important to fix the broken tokenizer in BioBERT base cased v1.2 distribution as by default it functions in uncased mode. The script takes care of this by forcing the toknizer into cased mode, but you can also use a working tokenizer configuration file from BioBERT base cased v1.1 repository – if needed.

# Installation

First step to fine-tune the models is to download the TEA datasets from the GitHub repository. This can be done easily by running the following command in the project root directory (creates the required TEA_datasets directory):

```bash
git clone https://github.com/tznurmin/TEA_datasets.git
```

Next, install the Python dependencies and you are good to go. Tested on Python 3.8.19, but these instructions should work with any modern Python version.

## Setting up virtual environment

Set up and activate a virtual environment in the project root:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Quick installation for Linux with existing CUDA support

To quickly set up the environment on Linux systems with existing CUDA (11.8) support, run the following commands:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers[torch] datasets evaluate seqeval
```

Newer CUDA versions will most probably also work. See [PyTorch website](https://pytorch.org) for instructions.

## More detailed installation instructions
For a comprehensive setup, a requirements.txt file is included for installing the necessary Python packages. Additionally, a flake.nix file is provided for optional system-level configuration (e.g. for system-level CUDA support).

Install the required Python packages by running:
```bash
pip install -r requirements.txt
```

Optionally, use the provided flake in a Nix system by running the following in the root directory:

```bash
nix develop
```

# Running the experiments

Use tea_ft.py to run the fine-tuning experiments once the dependencies are available and the datasets have been downloaded. The script will select the correct datasets based on the given arguments and automatically creates a log directory for the experimental results.

The fine-tuning script requires two mandatory arguments: --type (pathogens or strains) to select experiment type and --experiment (augmentation, strategy, mix1, mix2 or mix3) to select the experiment. 

There are a few optional arguments, such as the used random seed (--seed), the number of epochs (-epochs) and the used batch size (--batch_size). See --h for help when running the script.

For example, pathogens augmentation experiment is run with seed 42:

```bash
python tea_ft.py --type pathogens --experiment augmentation --seed 42
```

This will create log files from running the experiments into 'logs' directory.
