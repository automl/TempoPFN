# TempoPFN: Synthetic Pre-Training of Linear RNNs for Zero-Shot Time Series Forecasting

[![preprint](https://img.shields.io/static/v1?label=Paper&message=2509.26468&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.25502) [![GIFT-Eval](https://img.shields.io/badge/%F0%9F%8F%86%20GIFT--Eval-Leaderboard-0078D4)](https://huggingface.co/spaces/Salesforce/GIFT-Eval) [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model_Repo-FFD21E)](https://huggingface.co/AutoML-org/TempoPFN) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/automl/TempoPFN/blob/main/LICENSE)

---

**TempoPFN** introduced in [TempoPFN: Synthetic Pre-Training of Linear RNNs for Zero-Shot Time Series Forecasting](https://arxiv.org/abs/2510.25502), is a univariate time series foundation model pretrained **entirely on synthetic data**. It delivers top-tier zero-shot forecasting accuracy while remaining fully reproducible and free from real-data leakage.

Built on a **Linear RNN (GatedDeltaProduct)** backbone, TempoPFN performs end-to-end forecasting without patching or windowing. Its design enables fully parallelizable training and inference while maintaining stable temporal state-tracking across long sequences. The GatedDeltaProduct architecture is based on [DeltaProduct](https://arxiv.org/html/2502.10297v3), extended with state-weaving for time series forecasting. For detailed information about the architecture and custom modifications, see [`src/models/gated_deltaproduct/README.md`](src/models/gated_deltaproduct/README.md).

This repository includes the [**pretrained 38M parameter model**](https://www.dropbox.com/scl/fi/mqsni5lehooyaw93y3uzq/checkpoint_38M.pth?rlkey=3uyehvmtted02xkha24zgpzb6&st=seevsbkn&dl=0), all training and inference code, and the **complete synthetic data generation pipeline** used for pretraining.

## ✨ Why TempoPFN?

* **High Performance, No Real Data:** Achieves top-tier competitive results on **GIFT-Eval, outperforming all existing synthetic-only approaches** and **surpassing the vast majority of models trained on real-world data**. This ensures full reproducibility and eliminates benchmark leakage.
* **Parallel and Efficient:** The linear recurrence design enables full-sequence parallelization. This gives us the best of both worlds: the linear efficiency of an RNN, but with the training parallelism of a Transformer.
* **Open and Reproducible:** Includes the full synthetic data pipeline, configurations, and scripts to reproduce training from scratch.  
* **State-Tracking Stability:** The GatedDeltaProduct recurrence and *state-weaving* mechanism preserve temporal continuity and information flow across long horizons, improving robustness without non-linear recurrence.

![TempoPFN Overview](https://iili.io/KDCHpou.png)

## ⚙️ Installation

This repository includes all training and inference code and the **complete synthetic data generation pipeline** used for pretraining.

The **pretrained 38M parameter model** is hosted on our **[Hugging Face repository](https://huggingface.co/AutoML-org/TempoPFN)**.

## 🚀 Get the Model & Quick Start

The easiest and recommended way to get the model, inference code, and weights is to clone our **[Hugging Face repository](https://huggingface.co/AutoML-org/TempoPFN)**.

```bash
# 1. Install Git LFS (if you haven't already)
# On Ubuntu: sudo apt-get install git-lfs
# On macOS: brew install git-lfs
git lfs install

# 2. Clone the Hugging Face repository
git clone https://huggingface.co/AutoML-org/TempoPFN
cd TempoPFN

# 3. Set up the environment
python3.12 -m venv venv & source venv/bin/activate
export PYTHONPATH=$PWD

# 4. Install PyTorch version matching your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 5. Install dependencies
pip install .
pip install .[dev]

# 4. Run the Quick Start Script 
python examples/quick_start_tempo_pfn.py

# 5. Alternatively, you can run the Notebook version
jupyter notebook examples/quick_start_tempo_pfn.ipynb
```

### Hardware & Performance Tips

**GPU Required:** Inference requires a CUDA-capable GPU with a matching PyTorch version installed. Tested on NVIDIA A100/H100.

**First Run:** The first inference for a new sequence length will be slow due to Triton kernel compilation. Subsequent runs will be fast.

**Cache Tip:** If using a network filesystem, prevent slowdowns by routing caches to a local directory (like `/tmp`) *before* running:
```bash
LOCAL_CACHE_BASE="${TMPDIR:-/tmp}/tsf-$(date +%s)"
mkdir -p "${LOCAL_CACHE_BASE}/triton" "${LOCAL_CACHE_BASE}/torchinductor"
export TRITON_CACHE_DIR="${LOCAL_CACHE_BASE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_CACHE_BASE}/torchinductor"

python examples/quick_start_tempo_pfn.py
```

## 🚂 Training

All training and model parameters are controlled via YAML files in `configs/`.  

```bash
# Single-GPU (Debug)
torchrun --standalone --nproc_per_node=1 src/training/trainer_dist.py --config ./configs/train.yaml

# Multi-GPU (e.g., 8 GPUs)
torchrun --standalone --nproc_per_node=8 src/training/trainer_dist.py --config ./configs/train.yaml
```

## 💾 Synthetic Data Generation

A core contribution of this work is our open-source synthetic data pipeline, located in `src/synthetic_generation/`. It combines diverse generators with a powerful augmentation cascade.

**Generators Used:**

* **Adapted Priors:** ForecastPFN, KernelSynth, GaussianProcess (GP), and CauKer (Structural Causal Models).
* **Novel Priors:** SDE (a flexible regime-switching Ornstein-Uhlenbeck process), Sawtooth, StepFunction, Anomaly, Spikes, SineWave, and Audio-Inspired generators (Stochastic Rhythms, Financial Volatility, Network Topology, Multi-Scale Fractals).

You can easily generate your own data by installing the development dependencies and instantiating a generator wrapper. See `examples/generate_synthetic_data.py` for a minimal script, or inspect the generator code in `src/synthetic_generation/`.

## 🤝 License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details. This permissive license allows for both academic and commercial use.

## 📚 Citation

If you find TempoPFN useful in your research, please consider citing our paper:
```bibtex
@misc{moroshan2025tempopfn,
  title={TempoPFN: Synthetic Pre-Training of Linear RNNs for Zero-Shot Time Series Forecasting},
  author={Vladyslav Moroshan and Julien Siems and Arber Zela and Timur Carstensen and Frank Hutter},
  year={2025},
  eprint={2510.25502},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
