# Understanding and Improving Flaky Test Classification <img padding="10" align="right" src="https://www.acm.org/binaries/content/gallery/acm/publications/artifact-review-v1_1-badges/artifacts_evaluated_functional_v1_1.png" alt="ACM Artifacts Evaluated - functional v1.1" width="60" height="60"/> <img padding="10" align="left" src="https://www.acm.org/binaries/content/gallery/acm/publications/artifact-review-v1_1-badges/artifacts_available_v1_1.png" alt="ACM Artifacts Available v1.1" width="60" height="60"/>


# Purpose
This README accompanies the artifact for the paper "Understanding and Improving Flaky Test Classification." This work introduces a new dataset that combines flaky and non-flaky tests, resulting in a more realistic distribution of test categories. The paper also presents a classification approach using fine-tuned models to predict the category of each test.

The primary goal of this artifact is to provide access to the dataset, training and inference code, and scripts for interpreting model predictions by identifying key features with attribution scores.

We are applying for the "Available" and "Functional" badges. The dataset and code are publicly accessible via a Zenodo link, and all scripts are fully executable, enabling others to reproduce our results and extend the work.

# Provenance

The full artifact can be obtained from 
```shell
https://drive.google.com/file/d/1X0DQquXsu56lHclp-6ZxMVJ88tFlC_-p/view?usp=sharing
```
Or https://zenodo.org/records/15761937 from Zenodo in the form of a Docker image. Note that the size of our Docker image including the dataset is about 11GB, so please prepare sufficient disk space.

# Requirements

docker

NVIDIA GPU

Disk Space 11GB


# Hardware Requirement

We perform fine-tuning on a Linux machine equipped with a single NVIDIA RTX A5000 GPU, 125GB of RAM, and 48GB of GPU memory, utilizing CUDA version 12.0.

# Setup 
After downloading flakylens_latest.tar.gz from the Zenodo link, load it into Docker:

```shell
docker load < flakylens_latest.tar.gz
```

Ensure that your user is added to the docker group to interact with the Docker daemon without needing sudo. If you're not already a member of the docker group, please follow these instructions to run Docker commands as a non-root user: https://docs.docker.com/engine/install/linux-postinstall/.

To start a Docker container, use the following command:

```shell
docker run --gpus all -it --rm flakylens:latest
```

We evaluated our artifact on our end using Docker version 20.10.21, build 20.10.21-0ubuntu1~20.04.2.

# Dataset

Our dataset is available in ```dataset/FlakyLens/FlakyLens_dataset_with_nonflaky_indented.csv```
Our dataset contains 8,574 tests collected from 97 projects, spanning six categories — five types of flaky tests and one non-flaky category. Each row in the dataset includes the following columns:

id – A unique identifier for each test case.

project – The name of the project the test belongs to.

test_name – The name of the test method.

full_code – The complete source code of the test method.

label – The numeric label assigned to the test, ranging from 0 to 5 (where 0–4 represent different flaky categories and 5 indicates non-flaky).

category – The string name of the test category, such as "Async-Wait" or "Non-Flaky".

# Fine-Tuning

To reproduce the full pipeline from scratch, you may choose to fine-tune the model:
```shell
bash per_project_prediction.sh FlakyLens "train"
```

Fine-tuning takes significant time (typically 8–10 hours on a standard GPU setup). However, you do not need to fine-tune the model to run the remaining steps or evaluate the research questions. We provide the fine-tuned models in the models/ directory.  All RQ evaluations are based on them unless otherwise noted.

# Usage

This repository supports running four major research questions (RQs) from our study.

---

## RQ1: Run Prediction with FlakyLens

To evaluate the predictive performance of **FlakyLens**, run:

```bash
bash rq1.sh
```

This will print the precision, recall, and F1-score for each category to the console. The output will be comparable to the results found in ```results/FlakyLens_Result_Found_By_Author.csv```.
Estimated runtime: ~2–3 minutes.

## RQ2: Predicting Flaky Tests with Other LLMs
To evaluate other large language models (LLMs) such as Qwen for flaky test prediction, you must first obtain a Hugging Face access token and set it as an environment variable:

```shell
export HF_TOKEN=<your_huggingface_token>
bash rq2.sh
```

The raw output is saved to FlakyLens_Categorization_PerProject-result/qwen.csv, and the final summary will be printed to the console. The results are comparable to ../results/qwen_Result_Found_By_Author.csv.
Estimated runtime: ~50–55 minutes.

## RQ3: Token-Level Attribution Analysis

To calculate attribution scores for each token using gradient-based analysis, run:
```shell
bash per_project_prediction.sh FlakyLens "BERT" "calculate_gradient"
```
The attribution results will be stored in: ```Attributions_scores/```.
For each category, a separate CSV file is provided containing two columns: Token and Attribution_Score.

Estimated runtime: ~25–30 minutes.

## RQ4: Perturbation-Based Robustness Evaluation
To evaluate the model under code perturbations (e.g., deadcode injection), run:

```shell
bash rq4.sh
```
Estimated runtime: ~5 minutes.
=======

