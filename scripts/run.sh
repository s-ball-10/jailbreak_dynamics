#!/bin/bash

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

source .venv/bin/activate

# Extracting activations
.venv/bin/python3 ./analysis/extracting_activations.py

# Generating graphs
.venv/bin/python3 ./analysis/pca_graphs.py
.venv/bin/python3 ./analysis/similarity_transferability_vectors.py
.venv/bin/python3 ./analysis/plot_harmfulness_evolution.py

# Steering
.venv/bin/python3 ./src/components/normalize_vectors.py
.venv/bin/python3 ./analysis/steering.py

# Evaluation
.venv/bin/python3 ./evaluation/llama_guard.py #obtain all ASR scores (without steering)
.venv/bin/python3 ./evaluation/score_steering.py #obtain ASR scores (with steering)


# salloc --gres=gpu:a100:1 --partition=a100 --time=02:00:00

