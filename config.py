"""
config.py — Central configuration for CamouflageNet.

All hyperparameters are taken directly from the paper:
  "Improving ICS Security Through Honeynets and Machine Learning Techniques"
  Alnozami, Ananbeh, Kim (2025)
"""

# ─── K-Center (Algorithm 1) ────────────────────────────────────────────────────
K_CLUSTERS      = 45     # Optimal k, selected via Elbow Method (tested 10–100)
ALPHA_THRESHOLD = 1.5    # Anomaly multiplier: flag if d_min > alpha * r_max
                         # Grid-searched over {1.0, 1.25, 1.5, 1.75, 2.0}
RANDOM_SEED     = 42     # First center chosen at random with this seed

# ─── Feature Engineering ────────────────────────────────────────
TIME_WINDOW = 60          # Session window in seconds (T = 60 s)
FEATURE_DIM = 7           # [f_freq, f_vol, f_proto×3, f_cmd, f_err]

# Protocol one-hot encodings
PROTOCOL_ENCODINGS = {
    "modbus": [1, 0, 0],
    "dnp3":   [0, 1, 0],
    "http":   [0, 0, 1],
}

# Command severity weights (ICS-CERT classification used in paper)
# Low-risk reads → 0.1 ; high-risk writes/Force Single Coil → 0.9
CMD_WEIGHTS = {
    "read":       0.1,   # Modbus FC 0x03 – Read Holding Registers
    "coil_read":  0.2,   # Modbus FC 0x01 – Read Coils
    "diagnostic": 0.4,   # Modbus FC 0x08 – Diagnostics
    "write_reg":  0.7,   # Modbus FC 0x06 – Write Single Register
    "write_multi":0.8,   # Modbus FC 0x10 – Write Multiple Registers
    "force_coil": 0.9,   # Modbus FC 0x05 – Force Single Coil (high risk)
    "unknown":    0.5,
}

# ─── Dataset ───────────────────────────────────────────────────────
# 14-day capture; 1.2 M packets (85 % benign, 15 % malicious)
# Full-scale paper dataset: 1.2M sessions (85% benign, 15% malicious).
TOTAL_PACKETS    = 1_200_000
BENIGN_RATIO     = 0.85
MALICIOUS_RATIO  = 0.15

# Training phase: 24 h benign background (paper: 100 k sessions)
N_TRAIN_BENIGN   = 100_000
# Validation: 10 % of training, used for threshold grid search
N_VAL_BENIGN     = 10_000

# Attack-session counts (Table 6 in paper — full scale)
ATTACK_COUNTS = {
    "port_scanning":    52_400,
    "modbus_fuzzing":   38_700,
    "dos":              61_200,
    "command_injection":27_700,
}
TOTAL_ATTACKS = sum(ATTACK_COUNTS.values())   # 180 000 (paper: 180 000)

# Benign sessions in test set (maintain ~85/15 ratio)
N_TEST_BENIGN = int(TOTAL_ATTACKS * (BENIGN_RATIO / MALICIOUS_RATIO))

# ─── Network Topology (Table 2) ───────────────────────────────────────────────
N_NODES     = 45
N_PLCS      = 12
N_RTUS      = 8
N_HMIS      = 5
N_HONEYPOTS = 20          # Honeypot nodes in CamouflageNet
N_REAL      = N_NODES - N_HONEYPOTS  # 25 real production nodes

# ─── CamouflageNet Reconfiguration──────────────────────
ROTATION_INTERVAL  = 120   # Honeypot IP/MAC rotates every 120 s
TTI_TARGET_RATIO   = 0.80  # Attacker must map 80 % of active nodes
N_MONTE_CARLO      = 100   # Monte Carlo runs for TTI averaging

# Expected TTI from paper (used for validation)
STATIC_TTI_EXPECTED     = 121   # seconds (static network baseline
CAMOUFLAGENET_TTI_EXPECTED = 404  # seconds (with CamouflageNet)
TTI_STD_EXPECTED        = 34    # ± std-dev

# ─── Elbow Method ─────────────────────────────────────────────────────────────
ELBOW_K_MIN = 10
ELBOW_K_MAX = 100

# ─── Single-seed expected results (Table 2 seed=42 column, ablation) ─────────
# Used by run_experiment.py and run_ablation.py.
# F1 is computed from the 85 %/15 % test set (1.02 M benign, 180 K attacks).
# Precision = DR*0.15 / (DR*0.15 + FPR*0.85); F1 = 2*P*R/(P+R).
EXPECTED_RESULTS = {
    "Proposed (k-Center)": {"DR": 96.5, "FPR": 1.8,  "F1": 0.93},
    "Standard k-Means":    {"DR": 91.4, "FPR": 3.5,  "F1": 0.87},
    "Static IDS (Snort)":  {"DR": 84.2, "FPR": 2.1,  "F1": 0.86},
    "Random Forest":       {"DR": 98.1, "FPR": 1.2,  "F1": 0.96},
    "LSTM Network":        {"DR": 97.3, "FPR": 1.5,  "F1": 0.95},
    "Autoencoder":         {"DR": 93.8, "FPR": 2.8,  "F1": 0.89},
}

# ─── Multi-seed expected results (Table 2, mean ± std columns) ────────────────
# Produced by experiments/run_multi_seed.py (seeds 1–10).
# These are the authoritative values reported in the paper.
# Wilcoxon signed-rank: W=0, p=0.002, Cliff's delta=±1.00 for all metrics.
MULTI_SEED_EXPECTED = {
    "Proposed (k-Center)": {
        "DR":  {"mean": 96.60, "std": 0.28},
        "FPR": {"mean":  1.77, "std": 0.09},
        "F1":  {"mean":  0.931, "std": 0.007},
    },
    "Standard k-Means": {
        "DR":  {"mean": 91.40, "std": 0.54},
        "FPR": {"mean":  3.50, "std": 0.26},
        "F1":  {"mean":  0.870, "std": 0.007},
    },
    # Wilcoxon (k-center vs k-means, two-tailed, n=10 paired runs)
    "Wilcoxon": {
        "DR":  {"W": 0, "p": "0.002", "cliff_delta":  1.00},
        "FPR": {"W": 0, "p": "0.002", "cliff_delta": -1.00},
        "F1":  {"W": 0, "p": "0.002", "cliff_delta":  1.00},
    },
}

# ─── Ablation Study (Table 5) ─────────────────────────────────────────────────
ABLATION_EXPECTED = {
    # F1 values consistent with the 85 %/15 % test-set class balance (paper Table 5).
    "Full System":                        {"DR": 96.5, "FPR": 1.8, "F1": 0.93, "TTI": 404},
    "– CamouflageNet (static honeypot)":  {"DR": 95.1, "FPR": 1.9, "F1": 0.92, "TTI": 121},
    "– ML Engine (manual review)":        {"DR": 78.3, "FPR": 4.7, "F1": 0.76, "TTI": 404},
    "– IPS Feedback Loop":                {"DR": 96.5, "FPR": 1.8, "F1": 0.93, "TTI": 215},
    "Replace k-Center with k-Means":      {"DR": 91.4, "FPR": 3.5, "F1": 0.87, "TTI": 404},
}

# ─── Miscellaneous ────────────────────────────────────────────────────────────
RESULTS_DIR   = "results"
FIGURES_DIR   = "results/figures"
MODELS_DIR    = "results/models"
