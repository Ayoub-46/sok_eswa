# SoK: Understanding Backdoor Attacks & Defenses in Federated Learning

> **Official code repository for the paper submitted to *Expert Systems With Applications* (Elsevier, ESWA).**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ESWA](https://img.shields.io/badge/Submitted%20to-ESWA%20%7C%20Elsevier-orange)](https://www.journals.elsevier.com/expert-systems-with-applications)

---

## 📖 Overview

This repository provides a **unified, modular framework** for systematically evaluating backdoor attacks and their corresponding defenses in Federated Learning (FL). It is the reference implementation accompanying our SoK (*Systematization of Knowledge*) paper, and is designed to support **reproducible, fair, and extensible** experimentation across a wide variety of threat models, aggregation strategies, and datasets.

The framework abstracts away the boilerplate of FL simulation so that researchers can focus on the attack/defense logic itself. All experiments are fully driven by **YAML configuration files**, requiring no code changes to reproduce any result from the paper.

---

## 👥 Authors

| Name | Role |
|---|---|
| **Ahmed Ayoub Bellachia** | Corresponding Author |
| Mouhamed Amine Bouchiha | Co-Author |
| Yacine Ghamri-Doudane | Co-Author |

---

## ✨ Key Features

- **6 backdoor attacks** implemented under a common interface
- **7 defenses** implemented and ready to plug in
- **2 aggregation algorithms** (FedAvg, FedOpt)
- **9 datasets** spanning vision, NLP, and federated benchmarks
- Fully **YAML-configurable** — reproduce any experiment with a single command
- Clean modular architecture under `src/` for easy extension
- Designed for **reproducible research**: fixed seeds, logged metrics, structured outputs

---

## 🗂️ Repository Structure

```
sok_eswa/
│
├── main.py                  # Entry point — parses config and launches experiment
├── requirements.txt         # Python dependencies
│
└── src/
    └── experiment/
        └── runner.py        # Core FL simulation loop (FederatedExperiment)
    └── attacks/             # Backdoor attack implementations
    └── defenses/            # Defense implementations
    └── aggregation/         # FL aggregation algorithms (FedAvg, FedOpt)
    └── data/                # Dataset loaders and federated partitioning
    └── models/              # Neural network architectures
    └── utils/               # Logging, metrics, seeding utilities
```

---

## ⚔️ Implemented Attacks

| Attack | Reference |
|---|---|
| **IBA** | Nguyen et al. |
| **Neurotoxin** | Zhang et al. |
| **Model Replacement (MR)** | Bagdasaryan et al. |
| **A3FL** | Zhang et al. |
| **3DFed** | Li et al. |
| **DarkFed** | Li et al. |

---

## 🛡️ Implemented Defenses

| Defense | Reference |
|---|---|
| **Krum** | Blanchard et al. |
| **Flame** | Nguyen et al. |
| **Deepsight** | Rieger et al. |
| **Clip + DP** | Sun et al. |
| **LeadFL** | Zhu et al. |
| **Trimmed-Mean** | Yin et al. |
| **Median** | Yin et al. |

---

## 📦 Supported Datasets

| Dataset | Domain |
|---|---|
| **MNIST** | Vision — digit recognition |
| **FEMNIST** | Vision — federated handwriting |
| **CIFAR-10** | Vision — object classification |
| **CIFAR-100** | Vision — fine-grained classification |
| **GTSRB** | Vision — traffic sign recognition |
| **ImageNet** | Vision — large-scale classification |
| **Flwr-Shakespeare** | NLP — next-character prediction |
| **20NewsGroups** | NLP — topic classification |
| **Sentiment140** | NLP — sentiment analysis |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ayoub-46/sok_eswa.git
cd sok_eswa
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run an experiment

All experiments are launched via `main.py` with a path to a YAML configuration file:

```bash
python main.py --config path/to/your/config.yaml
```

---

## ⚙️ Configuration

Experiments are fully controlled by YAML files. Below is a **minimal baseline configuration** (no attack, no defense — clean FL run):

```yaml
# configs/baseline.yaml

federated_learning:
  rounds: 100
  num_clients: 100
  clients_per_round: 10
  aggregation: FedAvg          # Options: FedAvg | FedOpt

model:
  architecture: resnet18        # Adjust per dataset

dataset:
  name: CIFAR-10
  iid: false                    # Use non-IID partitioning
  num_classes: 10

attack:
  name: none                    # No attack — baseline run

defense:
  name: none                    # No defense — baseline run

experiment:
  seed: 42
  output_dir: results/
  log_every: 10
```

To run an attack/defense experiment, simply replace the `attack.name` and `defense.name` fields with any of the supported options (e.g., `neurotoxin`, `flame`).

---

## 📊 Metrics

The framework logs the following metrics per round:

- **Main Task Accuracy (MTA)** — accuracy on the clean global test set
- **Attack Success Rate (ASR)** — fraction of poisoned inputs classified as the target label
- Per-round client update statistics for analysis

Results are saved in structured files under the configured `output_dir`.

---

## 🔧 Extending the Framework

### Adding a new attack

1. Create a new file in `src/attacks/`.
2. Subclass the base `Attack` interface and implement the `poison()` method.
3. Register the attack name in the experiment config loader.

### Adding a new defense

1. Create a new file in `src/defenses/`.
2. Subclass the base `Defense` interface and implement the `aggregate()` method.
3. Register the defense name in the experiment config loader.

---

## 📋 Requirements

```
giotto_tda
hdbscan
numpy
persim
PyYAML
scikit_learn
scipy
torch
torchvision
datasets
pandas
tqdm
```

Python **3.8+** and PyTorch **2.x** are recommended. GPU execution is supported automatically when a CUDA device is available.

---

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{bellachia2025sok,
  title     = {SoK: Understanding Backdoor Attacks \& Defenses in Federated Learning},
  author    = {Bellachia, Ahmed Ayoub and Bouchiha, Mouhamed Amine and Ghamri-Doudane, Yacine},
  journal   = {Expert Systems With Applications},
  publisher = {Elsevier},
  year      = {2025},
  note      = {Under review}
}
```

---

## 📬 Contact

For questions, please open a GitHub issue or contact the corresponding author:

**Ahmed Ayoub Bellachia** — via GitHub Issues on this repository.

---

## 📜 License

This project is released under the [MIT License](LICENSE).