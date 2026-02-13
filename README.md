# Error Amplification Limits ANN-to-SNN Conversion in Visual Continuous Control

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)

This repository is the official implementation for the paper **"Error Amplification Limits ANN-to-SNN Conversion in Visual Continuous Control"**.

The project demonstrates that while Artificial-to-Spiking Neural Network (ANN-to-SNN) conversion works well for image classification, it suffers severe performance degradation in vision-based continuous control tasks due to **spatiotemporal error amplification**. We propose **Cross-Step Residual Potential Initialization (CRPI)**, a simple, training-free method to mitigate this issue and recover near-perfect performance.

![Illustration of the Problem and Solution](https://ar5iv.labs.arxiv.org/html/2204.13271/assets/figure/Conversion.png)


## 📌 Core Idea

In continuous control, an agent interacts with an environment over many sequential steps. Minor quantization errors introduced during ANN-to-SNN conversion are not independent; they develop **positive temporal correlations**. This means errors at one time step reinforce similar errors at the next, causing the agent's state trajectory to drift further and further away from the optimal path, leading to catastrophic failure.

**CRPI** solves this by initializing the membrane potential of spiking neurons at the start of a new decision step using the **residual membrane potential** left over from the previous step. This cross-step memory breaks the error correlation and stabilizes long-horizon behavior.

## 🚀 Features

- **Reproduction of DrQ-v2**: A sample-efficient off-policy reinforcement learning algorithm for pixel-based control.
- **Modular ANN-to-SNN Conversion**: Supports multiple neuron models including:
    - Integrate-and-Fire (IF)
    - Signed Neuron with Memory (SNM)
    - Multi-Threshold (MT) Neurons
    - Differential Coding (DC)
- **Plug-and-Play CRPI**: The core `CRPI` mechanism can be easily integrated into any of the above conversion methods without retraining.
- **DeepMind Control Suite (DMC) Benchmark**: Evaluation on six challenging vision-based continuous control tasks.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/ann-snn-continuous-control.git
    cd ann-snn-continuous-control
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    The `requirements.txt.txt` file contains all necessary packages.
    ```bash
    pip install -r requirements.txt.txt
    ```

## ▶️ Usage

The workflow consists of two main phases: **Training** an ANN policy and **Converting/Evaluating** it as an SNN.

### 1. Training the agent

Use `train.py` to train a DrQ-v2 agent on a DMC task. The default configuration is for the `cheetah_run` task.

```bash
python train.py task=quadruped_walk
```
### 2. Converting the ANN to SNN
Rename and move the model to correct dictionary,
```sh
python convert.py env_name=quadruped_walk SNN_ts=32
```