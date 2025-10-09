# Assignment 2: Fine-Tuning a Genomic Foundation Model on 3D Genome Prediction

The [`DNALongBench`](https://github.com/ma-compbio/DNALONGBENCH) benchmark suite introduced a set of challenging problems in genomics that require understanding long-range dependencies in DNA. One of the most difficult tasks is predicting the 3D folding of chromatin (a "contact map") from a 1D DNA sequence. The paper showed that standard CNNs fail at this, while specialized models perform well, highlighting the need for powerful architectures.

In this assignment, you'll work with **Evo2**, a state-of-the-art foundation model for genomics. Your task is to fine-tune a linear prediction head on top of the pre-trained Evo2 model on the **Contact Map Prediction** task. You will learn to manage a complex deep learning environment, use Weights & Biases (`wandb`) to monitor your training runs, and visually analyze the model's predictions to understand what it has learned about the 3D genome.

## Environment Setup

The starting code and environment instructions are provided in the DNALongBench and Evo2 repositories.

**Prerequisites:** You will be provided with access to the necessary **NVIDIA H100 GPUs** for this assignment, as they are required to run the Evo2 codebase.

You will need to set up **two separate conda environments**.

1.  **DNALongBench Environment (for data loading & utilities)**

    ```bash
    conda create -n dnalongbench python=3.9 -y 
    conda activate dnalongbench
    git clone https://github.com/wenduocheng/DNALongBench.git
    cd DNALongBench
    pip install .
    ```

2.  **Evo2 Environment (for model fine-tuning)**

    ```bash
    conda create -n evo2 python=3.11 -y 
    conda activate evo2
    git clone --recurse-submodules git@github.com:ArcInstitute/evo2.git
    cd evo2
    pip install .
    pip install wandb
    ```

## Downloading Data & Pretrained Models

**Dataset:**
Please download the **Contact Map Prediction** dataset from the DNALongBench repository. To save time, we recommend downloading it once to a shared directory on the cluster.

**Pretrained Model:**
The pre-trained **Evo2** model weights will be downloaded and cached automatically the first time you run the fine-tuning script provided in the `DNALongBench/experiments/Evo2` folder.

## Finetuning & Monitoring with `wandb`

We have provided a skeleton script for fine-tuning. Your primary coding task is to instrument this script with Weights & Biases (`wandb`) to track the experiment. Please **do not change the hyperparameters** to ensure results are comparable.

After adding the `wandb` integration code to the `TODO` sections in `test/filetune_contact_map.py`, run the following command to fine-tune the model:

```bash
python test/filetune_contact_map.py
```

## Evaluation & Visualization

After fine-tuning, you will evaluate your best checkpoint and create a case-study visualization to interpret the model's predictions.

1.  First, run the evaluation script:

    ```bash
    python test/evaluate_contact_map.py
    ```

2.  Next, from your evaluation results, identify a single genomic region from the test set where your model achieved a very high SCC or PCC score.

3.  For this region, create a visualization figure similar to Figure 2 in the DNALongBench paper. Your figure should include two subplots:

      * The **Ground Truth** contact map.
      * Your **Finetuned Evo2 Prediction** contact map.
      * Be sure to label each subplot with the corresponding PCC and SCC score.

## Submit Your Results

Please compress the following into a single zip file for submission:

1.  **The fully filled code**. Submit only your modified python files (`test/filetune_contact_map.py` and `test/evaluate_contact_map.py`).
2.  **A PDF report**. The report should include:
      * A link to your public `wandb` project page.
      * Your case-study visualization figure.
      * A written analysis comparing your Evo2 results to the baseline performances reported in the DNALongBench paper (e.g., CNN and Akita). Discuss *why* Evo2 is so effective for this task.
