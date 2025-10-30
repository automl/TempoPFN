# TempoPFN: Synthetic Pre-Training of Linear RNNs for Zero-Shot Time Series Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2510.25502-b31b1b.svg)](https://arxiv.org/abs/2510.25502)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/automl/TempoPFN/blob/main/LICENSE)

---

**TempoPFN** introduced in [TempoPFN: Synthetic Pre-Training of Linear RNNs for Zero-Shot Time Series Forecasting](https://arxiv.org/abs/2510.25502), is a univariate time series foundation model pretrained **entirely on synthetic data**. It delivers top-tier zero-shot forecasting accuracy while remaining fully reproducible and free from real-data leakage.

Built on a **Linear RNN (GatedDeltaProduct)** backbone, TempoPFN performs end-to-end forecasting without patching or windowing. Its design enables fully parallelizable training and inference while maintaining stable temporal state-tracking across long sequences.

This repository includes the [**pretrained 35M parameter model,**](https://www.dropbox.com/scl/fi/5vmjr7nx9wj9w1vl2giuv/checkpoint.pth?rlkey=qmk08ojp7wj0l6kpm8hzgbzju&st=dyr07d00&dl=0), all training and inference code, and the **complete synthetic data generation pipeline** used for pretraining.

## ✨ Why TempoPFN?

* **High Performance, No Real Data:** Achieves top-tier competitive results on **GIFT-Eval, outperforming all existing synthetic-only approaches** and **surpassing the vast majority of models trained on real-world data**. This ensures full reproducibility and eliminates benchmark leakage.
* **Parallel and Efficient:** The linear recurrence design enables full-sequence parallelization. This gives us the best of both worlds: the linear efficiency of an RNN, but with the training parallelism of a Transformer.
* **Open and Reproducible:** Includes the full synthetic data pipeline, configurations, and scripts to reproduce training from scratch.  
* **State-Tracking Stability:** The GatedDeltaProduct recurrence and *state-weaving* mechanism preserve temporal continuity and information flow across long horizons, improving robustness without non-linear recurrence.


![TempoPFN Overview](https://iili.io/KlUjfcP.png)

## ⚙️ Installation

```bash
git clone https://github.com/automl/TempoPFN.git
cd TempoPFN
python -m venv venv && source venv/bin/activate

# 1. Install PyTorch first (see PyTorch website for your specific CUDA version)
# Example for CUDA 12.6:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. Install TempoPFN and all other dependencies
pip install .
export PYTHONPATH=$PWD
```

## 🚀 Quick Start: Run the Demo



**Prerequisites:**
* You must have a **CUDA-capable GPU** with a matching PyTorch version installed.
* You have run `export PYTHONPATH=$PWD` from the repo's root directory (see Installation).

### 1. Run the Quick Start Script

Run a demo forecast on a synthetic sine wave:
```bash
python examples/quick_start_tempo_pfn.py
```

### 2. Run with a Local Checkpoint

If you have already downloaded the model (e.g., to `models/checkpoint.pth`), you can point the script to it:
```bash
python examples/quick_start_tempo_pfn.py --checkpoint models/checkpoint.pth
```

### 3. Run the Notebook version

```bash
jupyter notebook examples/quick_start_tempo_pfn.ipynb
```

### Hardware & Performance Tips

**GPU Required:** Inference requires a CUDA-capable GPU. Tested on NVIDIA A100/H100.

**Triton Caches:** To prevent slowdowns from writing caches to a network filesystem, route caches to a local directory (like `/tmp`) before running:
```bash
LOCAL_CACHE_BASE="${TMPDIR:-/tmp}/tsf-$(date +%s)"
mkdir -p "${LOCAL_CACHE_BASE}/triton" "${LOCAL_CACHE_BASE}/torchinductor"
export TRITON_CACHE_DIR="${LOCAL_CACHE_BASE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_CACHE_BASE}/torchinductor"

python examples/quick_start_tempo_pfn.py
```

## 🚂 Training


### Single-GPU Training (for debugging)
```bash
torchrun --standalone --nproc_per_node=1 src/training/trainer_dist.py --config ./configs/train.yaml
```

### Multi-GPU Training (Single-Node)

This example uses 8 GPUs. The training script uses PyTorch DistributedDataParallel (DDP).
```bash
torchrun --standalone --nproc_per_node=8 src/training/trainer_dist.py --config ./configs/train.yaml
```

### Configuration

All training and model parameters are controlled via YAML files in `configs/` (architecture, optimizers, paths).  

## 💾 Synthetic Data Generation

A core contribution of this work is our open-source synthetic data pipeline, located in `src/synthetic_generation/`. It combines diverse generators with a powerful augmentation cascade.

**Generators Used:**

* **Adapted Priors:** ForecastPFN, KernelSynth, GaussianProcess (GP), and CauKer (Structural Causal Models).
* **Novel Priors:** SDE (a flexible regime-switching Ornstein-Uhlenbeck process), Sawtooth, StepFunction, Anomaly, Spikes, SineWave, and Audio-Inspired generators (Stochastic Rhythms, Financial Volatility, Network Topology, Multi-Scale Fractals).

You can easily generate your own data by instantiating a generator wrapper. See `examples/generate_synthetic_data.py` for a minimal script, or inspect the generator code in `src/synthetic_generation/`.

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
