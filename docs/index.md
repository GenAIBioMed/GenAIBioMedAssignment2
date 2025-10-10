# Assignment 2: Fine-Tuning Evo2 on 3D Genome Prediction

## Overview

The [`DNALongBench`](https://www.biorxiv.org/content/10.1101/2025.01.06.631595v1.full.pdf) paper introduced a set of challenging problems in genomics that require understanding long-range dependencies in DNA. One of the most difficult tasks is predicting the 3D folding of chromatin (a "contact map") from a 1D DNA sequence of over 1 million base pairs.

![DNALongBench Overview](https://github.com/ma-compbio/DNALONGBENCH/raw/main/Figure1.v3.png)

In this assignment, you will fine-tune **Evo2** (7B), a state-of-the-art genomic foundation model, on this **Contact Map Prediction** task.

You will learn to manage a complex deep learning environment, use Weights & Biases (`wandb`) for experiment tracking, and visually analyze the model's predictions to interpret what it has learned.

---

## 1. Environment Setup

**IMPORTANT:** This assignment requires access to **NVIDIA H100 GPUs** (Since Evo2 requires device compute capability 8.9 or higher required for FP8 execution). 

You must first request a GPU node on the cluster *before* setting up the Conda environments. This ensures that Conda can correctly detect the CUDA toolkit.

**Example: Requesting a GPU Node (SLURM)**
```bash
srun --partition=GPU-shared --gres=gpu:h100-80:1 --time=1:00:00 --account=cis250160p --pty bash
```

### **Evo2 Environment (for model fine-tuning)**


You will need to run the following commmands everytime you attempt to train the model.
```bash
module load anaconda3/2024.10-1
module load cuda/12.4.0
```

Create a conda evnvironment named `evo2` with Python 3.12 and install the required packages.

```bash
conda create -n evo2 python=3.12 -y
conda activate evo2

conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation

pip install evo2
pip install natsort
pip install tensorflow==2.17.0
pip install wandb
pip install matplotlib scipy
```

### **Fixing a version issue with glibc (follow these instructions carefully)**

After setting up the environment, you will need to fix a dependency issue.
Flash-attn package requires `GLIBC_2.32`, while PSC has the version 2.28.

Follow the following steps mentioned in this [github comment](https://github.com/Dao-AILab/flash-attention/issues/1708#issuecomment-3283420504): 

1. Build polyfill-glibc with feat/single-threaded support
2. Get the path to flash-attn's .so lib
3. Get your current GLIBC version
4. Patch flash-attn's .so lib

After you perform these steps, in the evo2 environement try running this command `python3 -c "import flash_attn"`. If it succeeds without error, you are good to go! Now you have a conda environment that supports Evo2 and Flash-attn! 

If you are having trouble at this step, make a post on edstem.

-----

## 2\. Data paths and Huggingface cache dir

### Huggingface Cache Directory (for model weights)

We have downloaded the Evo2 model weight on PSC. Therefore run this command to make sure huggingface looks at the correct location for the model weights. Otherwise it will download the weights to your home directory which has limited space.

```bash
export HF_HOME=/ocean/projects/cis250160p/rhettiar
echo 'export HF_HOME=/ocean/projects/cis250160p/rhettiar' >> ~/.bashrc;
```

This line adds the environment variable to your `.bashrc` file, so it is set automatically in future sessions. After completing the assignment, remove this line from your `.bashrc`.

### Dataset Path

We have downloaded the **Contact Map Prediction** dataset and have placed it in the shared directory (`
/ocean/projects/cis250160p/rhettiar/contact_map_prediction/`). Try navigating to that directory (`cd`) to check if you can see the dataset. 

The scripts that you will use for fintuning already has this path set as the default data path. Therefore, if you plan to run the scripts on a different cluster make sure to edit `data_path`.

If you are running on PSC, the dataloader will automatically use this path to load the data. If you are running on a different machine, download the dataset from [this link](https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/AZM25S)
Note: For reduced train time, we are only using `test-0.tfr`  `train-0.tfr`  `valid-0.tfr` from this contact map dataset.

-----

## 3\. Finetuning & Evaluation


Clone this assignemnt 2 repository to your working directory.

```bash
git clone https://github.com/GenAIBioMed/GenAIBioMedAssignment2
```

### a) Integrate with `wandb` (for experiment tracking)
Your first task is to edit the finetuning code (`finetune_contact_map.py`). Fill in the sections marked `TODO` to integrate Weights & Biases (`wandb`) for experiment tracking. Then run finetuning with `python finetune_contact_map.py`. You can refer to this documentation when learning about wandb [Getting Started with Weights & Biases](https://docs.wandb.ai/quickstart/).

### b) Run Finetuning & Evaluation

Activate your `evo2` environment and run the fine-tuning script. The script is configured to use the evo2_7b model by default. The lines below loads cuda and sets the huggingface cache path to the shared folder.

```bash
conda activate evo2
module load cuda/12.4.0
export HF_HOME=/ocean/projects/cis250160p/rhettiar
python finetune_contact_map.py
```

After training is complete, use your best model checkpoint to run the evaluation script (this will saved model predictions and groundtruth as .npy files).

```bash
conda activate evo2
module load cuda/12.4.0
export HF_HOME=/ocean/projects/cis250160p/rhettiar
python evaluate_contact_map.py
```

#### Visualize Predictions & Analyze Performance

The `evaluate_contact_map.py` script saves the raw prediction and target data. You will create a new Python script (e.g., `analyze_performance.py`) to process these files.

Your script must perform two key tasks:

1.  **Calculate Overall Performance:**
    * Load the `pred.npy` and `target.npy` files.
    * Iterate through every sample in the test set. For each sample, calculate the Pearson (PCC) between the predicted and the ground truth contact map.
    * Compute the **average PCC** across the *entire* test set. These values represent your model's overall performance.

2.  **Generate a Representative Visualization:**
    * After calculating all scores, find a single genomic region from the test set that demonstrates good performance (e.g., its score is near or above the average).
    * Generate a plot for this single example. The figure should contain two subplots: the **Ground Truth** map and your **Finetuned Prediction**.
    * Use `matplotlib.pyplot.imshow` to display the 50x50 matrices.
    * Clearly label each subplot with the specific PCC for that individual example.

---

## 4. Analysis & Submission

Compress the following into a single **zip file** for submission.

1.  **Modified Python Files:**
    * `finetune_contact_map.py` (with `wandb` integration).
    * `analyze_performance.py` (the script you wrote for calculation and visualization).

2.  **PDF Report:** Your report must include:
    * Report the final loss you achieve.
    * A link to your public `wandb` [experiment's report](https://docs.wandb.ai/guides/reports/create-a-report/) showing your training curves.
    * **Overall Performance Metrics:** State the **average PCC** you calculated across the entire test set.
    * **Representative Visualization:** Include the visualization figure you generated for a single, high-performing example. Ensure the subplots are clearly labeled with their specific scores.
    * **Written Analysis:** Your analysis should address the following points in a few short paragraphs:
        1.  **Performance Comparison:** Based on your average scores, does your fine-tuned Evo2 model achieve better or worse performance than the CNN baseline reported in the DNALongBench [paper](https://www.biorxiv.org/content/10.1101/2025.01.06.631595v1.full.pdf)?
        2.  **Discussion:** Whether it performs better or worse, discuss potential reasons for this outcome. What could have we done in the finetuning script for better performance?