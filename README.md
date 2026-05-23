# A Connectionist Model of Psychological Refractory Period Effect

Repository for the Master's Thesis project titled "A Connectionist Model of Psychological Refractory Period Effect and Cognitive Control." The project investigates human information processing limitations using a simple neural network model of cognitive control.

This project replicates and extends **Simulation Study 3** from:

> Musslick, S., et al. (2023).  
> *On the Rational Boundedness of Cognitive Control: Shared Versus Separated Representations.*


---

## Project Structure
* prp/: Core library for LCA dynamics and PRP trial logic.
* scripts/: Command-line interfaces for training and analysis.
* output/: Default directory for models and generated figures.
* notebooks/: Exploratory visualization and toy simulations.
---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/acesmeci/prp_model.git
cd prp_model
```

### 2. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model
Train network instances using the online training script:
```bash
python -m scripts.train_model
```

### 4. Ensemble Simulation
Run PRP sweeps across multiple trained networks to generate aggregate data:
```bash
python -m scripts.run_prp_ensemble --ckpt_dir <checkpoint_path> --E 20 --workers 6
```

### 5. Visualization
Generate RT2 vs. SOA plots with empirical comparisons to Pashler (1994):
```bash
python -m scripts.plot_prp_ensemble --ckpt_dir <checkpoint_path> --add_pashler --align_pashler_rt
```

For an exact match to the four SOAs (50, 150, 300, 900 ms) reported in Pashler (1994), use:
```bash
python -m scripts.plot_prp_ensemble_pashler --ckpt_dir <checkpoint_path> --align_pashler_rt
```

## Reference
Musslick, S., et al. (2023).
On the Rational Boundedness of Cognitive Control: Shared Versus Separated Representations.
