# Assignment 2: Fine-Tuning Evo2 on 3D Genome Prediction

## Overview

The [`DNALongBench`](https://github.com/ma-compbio/DNALONGBENCH) paper introduced a set of challenging problems in genomics that require understanding long-range dependencies in DNA. One of the most difficult tasks is predicting the 3D folding of chromatin (a "contact map") from a 1D DNA sequence of over 1 million base pairs.

![DNALongBench Overview](https://github.com/ma-compbio/DNALONGBENCH/raw/main/Figure1.v3.png)

In this assignment, you will fine-tune **Evo2**, a state-of-the-art genomic foundation model, on this **Contact Map Prediction** task. You will learn to manage a complex deep learning environment, use Weights & Biases (`wandb`) for experiment tracking, and visually analyze the model's predictions to interpret what it has learned.



---

## 1. Environment Setup

**IMPORTANT:** This assignment requires access to **NVIDIA H100 GPUs**. You must first request a GPU node on the cluster *before* setting up the Conda environments. This ensures that Conda can correctly detect the CUDA toolkit.

**Example: Requesting a GPU Node (SLURM)**
```bash
srun --partition=GPU-shared --gres=gpu:h100-80:1 --time=0:20:00 --account=cis250160p --pty bash
```

### Evo2 Environment (for model fine-tuning)

```bash
conda create -n evo2 python=3.12 -y
conda activate evo2

conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation

pip install evo2
pip install natsort
pip install tensorflow==2.17.0
```

### Fixing a version issue with glibc (Follow instructions carefully)

After setting up the environment, you will need to fix a dependency issue.
Flash-attn package requires `GLIBC_2.32`, while PSC has the version 2.28.

Follow the following steps mentioned in this [github comment](https://github.com/Dao-AILab/flash-attention/issues/1708#issuecomment-3283420504): 

1. Build polyfill-glibc with feat/single-threaded support
2. Get the path to flash-attn's .so lib
3. Get your current GLIBC version
4. Patch flash-attn's .so lib

After you perform these steps, in the evo2 environement try running this command `python3 -c "import flash_attn"`. If it succeeds without error, you are good to go! If you are having trouble at this step, reach out to us on edstem.

-----

## 2\. Data paths and Huggingface cache dir

### Huggingface Cache Directory

We have already downloaded the Evo2 model weights. Therefore run this command to make sure huggingface looks at the correct location for the model weights. Otherwise it will download the weights to your home directory which has limited space.

```bash
echo 'export HF_HOME=/ocean/projects/cis250160p/rhettiar' >> ~/.bashrc;
```

### Huggingface Cache Directory

We have already downloaded the **Contact Map Prediction** dataset and have placed it in the shared directory (`
/ocean/projects/cis250160p/rhettiar/contact_map_prediction/`). 

If you are running on PSC, the dataloader will automatically use this path to load the data. If you are running on a different machine, download the dataset from [this link](https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/AZM25S)

**Example Path:**

```bash
# Create a data directory in your scratch space
mkdir -p /scratch/your_username/data/dnalongbench
# Download and unzip the data into the directory above
```

### Pretrained Model Cache

The Evo2 model weights will be downloaded automatically by the script. To avoid filling up your home directory, specify a cache location in your scratch space using an environment variable.

**Example:**

```bash
export TRANSFORMERS_CACHE="/scratch/your_username/cache"
```

-----

## 3\. Finetuning & Evaluation

**IMPORTANT:** All training and evaluation commands must be run from within the `evo2` repository directory.

First, navigate to the correct directory:

```bash
cd evo2
```

### a) Instrument with `wandb`

Your first task is to edit the `test/filetune_contact_map.py` script. Fill in the sections marked `TODO` to integrate Weights & Biases (`wandb`) for experiment tracking. Please **do not change the existing hyperparameters**.

### b) Run Finetuning

Activate your `evo2` environment and run the fine-tuning script. Remember to set your model cache path.

```bash
conda activate evo2
export TRANSFORMERS_CACHE="/scratch/your_username/cache"
python test/filetune_contact_map.py --data_path /scratch/your_username/data/dnalongbench/contact_map
```

### c) Run Evaluation

After training is complete, use your best model checkpoint to run the evaluation script.

```bash
conda activate evo2
export TRANSFORMERS_CACHE="/scratch/your_username/cache"
python test/evaluate_contact_map.py --data_path /scratch/your_username/data/dnalongbench/contact_map --checkpoint_path /path/to/your/best/checkpoint.pt
```

-----

## 4\. Analysis & Submission

Compress the following into a single **zip file** for submission.

1.  **Modified Python Files:**

      * `test/filetune_contact_map.py` (with `wandb` integration)
      * `test/evaluate_contact_map.py` (if you made any changes for analysis)

2.  **PDF Report:** Your report must include:

      * A link to your public `wandb` project page showing your training curves.
      * A **case-study visualization figure**. You will need to write your own plotting code (e.g., with Matplotlib) to generate this.
          * First, inspect your evaluation results to find a genomic region from the test set with a high Pearson (PCC) or Spearman (SCC) score.
          * Your figure should contain two subplots: the **Ground Truth** contact map and your **Finetuned Evo2 Prediction** for that region.
          * Label each subplot clearly with its PCC and SCC score.
      * A **written analysis** comparing your Evo2 results to the baselines in the DNALongBench paper (CNN, Akita). Discuss potential reasons for Evo2's effectiveness on this long-range dependency task.