**Paper:** *Improving ICS Security Through Honeynets and Machine Learning Techniques*
**Authors:** Obieda Ananbeh, Wala Alnozami
---

## Overview

This repository provides a complete implementation of every component described in the paper. The system proposes a hybrid SCADA security architecture that synergises:

1. **CamouflageNet** — a dynamic high-interaction honeynet that rotates IP/MAC addresses every 120 seconds to delay adversarial reconnaissance
2. **k-Center Clustering** (Algorithm 1) — an unsupervised ML engine that identifies zero-day attacks without labeled training data
3. **IPS Feedback Loop** — automatic blocking rules generated from cluster analysis

**Key results replicated:**
- Mean detection rate 96.60% ± 0.28% across 10 independent runs (Wilcoxon vs k-means: W=0, p=0.002, Cliff's δ=1.00)
- 234% increase in adversarial reconnaissance time (mean TTI: 121s → 404s)
- Silhouette coefficient: 0.78 ± 0.03

---

## Project Structure

```
camouflage_net/
├── main.py                          # Entry point — run all experiments
├── config.py                        # All hyperparameters from paper
├── requirements.txt
│
├── data/
│   └── traffic_generator.py         # Synthetic SCADA session generator
│
├── features/
│   └── feature_engineering.py       # 5-feature extraction + min-max normalisation
│
├── models/
│   ├── k_center.py                  # Algorithm 1 (CORE CONTRIBUTION)
│   ├── k_means_detector.py          # k-Means baseline
│   ├── snort_detector.py            # Signature-based IDS simulation
│   ├── random_forest_detector.py    # Supervised RF baseline
│   ├── autoencoder_detector.py      # Unsupervised autoencoder baseline
│   └── lstm_detector.py             # LSTM supervised baseline (optional PyTorch)
│
├── simulation/
│   ├── camouflage_net.py            # Dynamic IP/MAC reconfiguration
│   └── tti_simulator.py             # Monte Carlo TTI analysis (100 runs)
│
├── evaluation/
│   └── metrics.py                   # DR, FPR, F1, Silhouette, per-attack metrics
│
│
└── experiments/
    ├── run_experiment.py            # Single-seed detection comparison (seed=42)
    ├── run_multi_seed.py            # Multi-seed stability experiment (seeds 1–10)
    ├── run_ablation.py              # Ablation study
    ├── run_tti_analysis.py          # TTI analysis
    └── run_elbow.py                 # Elbow Method
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments
python main.py

# 3. Or run individual experiments
python main.py --exp main       # Detection comparison — single seed=42
python main.py --exp ablation   # Ablation study
python main.py --exp tti        # TTI analysis
python main.py --exp elbow      # Elbow Method

# 4. Multi-seed stability experiment
#    Reproduces — seeds 1–10, ~10× longer than main.
python experiments/run_multi_seed.py
#    Custom seed list:
python experiments/run_multi_seed.py --seeds 1 2 3 4 5
#    Quiet mode (no per-seed progress):
python experiments/run_multi_seed.py --quiet

# 5. Skip LSTM if PyTorch is not installed
python main.py --no-lstm
```

Results are saved to `results/` and figures to `results/figures/`.
The multi-seed experiment writes `results/multi_seed_results.json`.

---

## Feature Engineering

Each session event **xᵢ ∈ ℝ⁷** is constructed from raw log entries:

| Feature | Dimension | Description | Benign Range | Attack Range |
|---------|-----------|-------------|--------------|--------------|
| `f_freq` | 1 | Request frequency (req/60s window) | 1–5 | >50 (scanning) |
| `f_vol` | 1 | Total bytes transferred | 200–600 | Varies by attack |
| `f_proto` | 3 | One-hot: Modbus=[1,0,0], DNP3=[0,1,0], HTTP=[0,0,1] | Mostly Modbus | Mixed |
| `f_cmd` | 1 | Command severity weight (0.1=read, 0.9=Force Single Coil) | 0.1–0.2 | 0.7–0.9 (injection) |
| `f_err` | 1 | Error response rate | 0–0.02 | 0.3–0.85 (fuzzing) |

All features are min-max normalised to **[0, 1]** before clustering.

---

## k-Center Algorithm (Algorithm 1)

The greedy 2-approximation algorithm for the k-center problem (Gonzalez 1985):

```
Input:  V = {x₁, …, xₙ}, k
Output: Centers C, assignments A

1. Select c₁ arbitrarily; C ← {c₁}
2. While |C| < k:
     x* ← argmax_{x∈V} min_{c∈C} ‖x − c‖₂
     C  ← C ∪ {x*}
3. ∀ xᵢ: A(xᵢ) ← argmin_{c∈C} ‖xᵢ − c‖₂

Anomaly Detection:
  r_max ← max cluster radius (training phase)
  For each new session xₙₑw:
    d_min ← min_{c∈C} ‖xₙₑw − c‖₂
    if d_min > α · r_max → ANOMALY → forward to IPS
    else → Normal (assign to nearest cluster)
```

**Parameters:** k = 45, α = 1.5, seed = 42

---

## Experimental Setup

The paper uses **Mininet v2.3.0** to emulate a 45-node water treatment SCADA network. Since Mininet requires a Linux environment with root privileges, this implementation replicates the statistical properties of the dataset synthetically.

### To use Mininet (real deployment):
- **Mininet**: https://github.com/mininet/mininet
- Install: `sudo apt-get install mininet` or `pip install mininet`
- Documentation: http://mininet.org/

### Honeypot Core — Conpot
The paper builds CamouflageNet on **Conpot**, configured to emulate Siemens S7-200 PLCs and Modbus/TCP slaves.
- **Conpot GitHub**: https://github.com/mushorg/conpot
- Install: `pip install conpot`
- Documentation: https://conpot.readthedocs.io/

### Traffic Generation Tools
- **Ostinato** (background traffic): https://ostinato.org/
- **Metasploit Framework** (attack traffic): https://www.metasploit.com/
- **Nmap** (port scanning attacks): https://nmap.org/

---

## Real Datasets

The paper uses a synthetic testbed. For real-world ICS datasets to validate the approach:

### Primary Datasets
| Dataset | Description | URL |
|---------|-------------|-----|
| **SWaT** (Secure Water Treatment) | 11-day physical water treatment plant data, 36 attacks | https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/ |
| **BATADAL** | Battle of the Attack Detection Algorithms — water distribution | http://www.batadal.net/ |
| **WADI** | Water Distribution testbed with 123 attack scenarios | https://itrust.sutd.edu.sg/testbeds/water-distribution-wadi/ |
| **HAI** | Hardware-In-the-Loop Augmented ICS security dataset | https://github.com/icsdataset/hai |

### ICS-Specific PCAP Datasets
| Dataset | Description | URL |
|---------|-------------|-----|
| **4SICS ICS Lab PCAP** | Captures from 4SICS conference ICS lab | https://www.netresec.com/?page=PCAP4SICS |
| **ORNL Modbus Dataset** | Modbus traffic captures | https://www.sc.com/files/ORNL-Modbus-Dataset.zip |
| **UNSW-NB15** | General network intrusion dataset | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| **CIC-IDS2017** | CICIDS 2017 intrusion detection | https://www.unb.ca/cic/datasets/ids-2017.html |

### Accessing SWaT / WADI (iTrust Datasets)
These require a Data Access Agreement. Request access at:
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

---

## Baseline Models

| Method | Reference | Implementation |
|--------|-----------|----------------|
| Snort (Signature IDS) | https://www.snort.org/ | Rule-based simulation in `models/snort_detector.py` |
| Random Forest [7] | Alimi et al. (2021) — DOI:10.3390/su13179597 | `sklearn.ensemble.RandomForestClassifier` |
| LSTM Network [14] | Tama et al. (2022) — DOI:10.1007/s11831-022-09767-y | PyTorch LSTM in `models/lstm_detector.py` |
| Autoencoder [17] | Choi & Kim (2024) — DOI:10.3390/asi7020018 | `models/autoencoder_detector.py` |
| k-Means | scikit-learn 1.3 | `models/k_means_detector.py` |

---

## Hyperparameters

| Parameter | Value | How Selected |
|-----------|-------|-------------|
| k (clusters) | 45 | Elbow Method, tested k ∈ [10, 100] |
| α (threshold) | 1.5 | Grid search over {1.0, 1.25, 1.5, 1.75, 2.0} on 10% benign validation |
| Rotation interval | 120 s | CamouflageNet reconfiguration period |
| Time window T | 60 s | Session aggregation window |
| Random seed | 42 | First center selection |
| Training size | 100,000 | 24h benign background traffic |

---

## Expected Results

### Table 3: Detection Comparison
| Method | DR (%) | FPR (%) | F1 |
|--------|--------|---------|-----|
| **Proposed (k-Center)** | **96.5** | **1.8** | **0.93** |
| Standard k-Means | 91.4 | 3.5 | 0.87 |
| Static IDS (Snort) | 84.2 | 2.1 | 0.86 |

### Table 4: SOTA Comparison
| Method | Type | DR (%) | FPR (%) | F1 | Labels Required |
|--------|------|--------|---------|-----|----------------|
| **Proposed (k-Center)** | **Unsupervised** | **96.5** | **1.8** | **0.93** | **No** |
| Random Forest | Supervised | 98.1 | 1.2 | 0.96 | Yes |
| LSTM Network | Supervised | 97.3 | 1.5 | 0.95 | Yes |
| Autoencoder | Unsupervised | 93.8 | 2.8 | 0.89 | No |

### Table 5: Ablation Study
| Configuration | DR | FPR | F1 | TTI (s) |
|---------------|-----|-----|-----|---------|
| Full System | 96.5 | 1.8 | 0.93 | **404** |
| – CamouflageNet (static) | 95.1 | 1.9 | 0.92 | 121 |
| – ML Engine | 78.3 | 4.7 | 0.76 | 404 |
| – IPS Feedback Loop | 96.5 | 1.8 | 0.93 | 215 |
| Replace k-Center with k-Means | 91.4 | 3.5 | 0.87 | 404 |

### Table 6: Per-Attack Category
| Attack | Count | DR (%) | FPR (%) | F1 |
|--------|-------|--------|---------|-----|
| Port Scanning | 52,400 | 99.2 | 0.4 | 0.99 |
| Modbus Fuzzing | 38,700 | 97.8 | 1.2 | 0.98 |
| DoS (Register Flooding) | 61,200 | 98.1 | 0.9 | 0.98 |
| Command Injection | 27,700 | 88.4 | 3.6 | 0.91 |
| **Overall (Weighted)** | **180,000** | **96.5** | **1.8** | **0.93** |

---

## Architecture Overview (Figure 1)

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    SCADA Security Architecture                    │
 │                                                                   │
 │  Attacker ─────────→ IDS ──[suspicious]──→ CamouflageNet        │
 │                        │                   (Honeypots × 20)      │
 │                        │[normal]                │                 │
 │                        ↓                        │ raw logs        │
 │              Production Network                 ↓                 │
 │              (PLCs, RTUs, HMIs)         ML Analysis Engine        │
 │                        ↑               (k-Center Clustering)      │
 │                        │                        │                 │
 │              IPS ←─[block rules]────────────────┘                │
 │           (Protector)                                             │
 └──────────────────────────────────────────────────────────────────┘
```

---

## Dependencies & Installation

```bash
# Required
pip install numpy scikit-learn matplotlib scipy

# Optional (for LSTM / PyTorch Autoencoder)
pip install torch

# All at once
pip install -r requirements.txt
```

**Python:** 3.9+ recommended
**scikit-learn:** ≥ 1.3 (matches paper's "Scikit-learn 1.3" specification)

---

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{alnozami2025camouflagenet,
  title   = {Improving {ICS} Security Through Honeynets and Machine Learning Techniques},
  author  = {},
  journal = {},
  year    = {2026},
}
```

---

## License

This implementation is provided for research and educational purposes.
