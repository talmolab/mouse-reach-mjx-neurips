 # mouse-reach-mjx-neurips

 <img width="1589" height="326" alt="arm_img" src="https://github.com/user-attachments/assets/69377199-05ec-48ce-bb82-c9438e71a151" />

High-throughput imitation learning for a musculoskeletal mouse forelimb model in **JAX-accelerated MuJoCo-MJX**.

This repository reproduces all figures and analyses from:

**Leonardis, E., Nagamori, A., Yang, Y., Park, J., Saunders, H., Azim, E., Pereira, T. D. (Accepted).**
*Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics.*
NeurIPS 2025 Workshop: *Data on the Brain & Mind – Concrete Applications of AI to Neuroscience and Cognitive Science*, San Diego, CA.

The project depends on **track-mjx**, which provides the imitation-learning infrastructure and high-throughput GPU rollouts:
[https://github.com/talmolab/track-mjx](https://github.com/talmolab/track-mjx)

---

## Quick Start

### 1. Install Dependencies

Install `track-mjx` and its requirements:

```bash
git clone https://github.com/talmolab/track-mjx
cd track-mjx
pip install -e .
```

Then install additional dependencies for analysis:

```bash
pip install jupyter matplotlib seaborn pandas h5py pyedm tqdm scikit-learn
```

Make sure MuJoCo-MJX is working correctly with GPU-accelerated JAX.

---

## 2. Start With the Demo Notebook

The fastest way to understand the pipeline is to run the **batch rollout demo**:

▶ **`demo_batch_rollout_PCA_figures.ipynb`**

This notebook demonstrates:

* How to load a trained checkpoint
* How to run batched imitation rollouts in parallel
* How to generate rollout `.h5` files
* How to compute PCA embeddings of intention and decoder-layer activations
* How to visualize reach trajectories and neural representations

This provides a complete end-to-end walkthrough of the training outputs used in the paper.

---

## 3. Generate Rollout HDF5 Files

The demo notebook saves rollouts in the same format described in **Data Table 1** of the paper (frames, joint angles, latent activations, decoder activations, muscle activations, etc.).

These rollout files are the input for the EMG and nonlinear forecasting analyses.

---

## 4. Plot EMG Comparisons

Use the notebook:

▶ **`emg_figures.ipynb`**

This notebook:

* Loads the rollout `.h5` generated in the demo
* Loads aligned biological EMG (biceps and triceps)
* Compares simulated muscle activations against observed EMG
* Computes trial-by-trial and averaged activation plots
* Reproduces the EMG MAE comparisons and activation time-series figures from the paper

This corresponds to the EMG panels in **Figure 2** of the manuscript.

---

## 5. Nonlinear Forecasting with PyEDM (Sugihara Simplex Projection)

To reproduce the nonlinear dynamical forecasting results, use:

▶ **`pyedm_figures.ipynb`**

This notebook implements:

* Takens-delay embedding of joint angles and simulated actions
* Sugihara’s **simplex projection** method
* Forecasting of simulated muscle activations from joint kinematics
* Forecasting of real EMG from simulated actions + reference kinematics
* τ, embedding-dimension, and prediction-horizon sweeps
* The forecasting accuracy plots (Simplex ρ) in **Figure 3**

This reproduces the nonlinear forecasting analysis in the manuscript.

---

## Data Availability (External)

You can download the datasets described in the paper here:

* **Training clip data (registered mocap → reference trajectories):**
  [https://huggingface.co/datasets/talmolab/MIMIC-MJX/tree/main/data/mouse_arm](https://huggingface.co/datasets/talmolab/MIMIC-MJX/tree/main/data/mouse_arm)

* **Model checkpoints and rollout data:**
  [https://huggingface.co/talmolab/mouse-reach-mjx-neurips](https://huggingface.co/talmolab/mouse-reach-mjx-neurips)

* **EMG, trial indexes, and parameter search results:**
  [https://huggingface.co/datasets/talmolab/mouse-reach-mjx-neurips](https://huggingface.co/datasets/talmolab/mouse-reach-mjx-neurips)

---

## Reproducing the Figures

Run the following notebooks in order:

1. **`demo_batch_rollout_PCA_figures.ipynb`**
   Generates rollouts + PCA visualizations.

2. **`emg_figures.ipynb`**
   Recreates EMG comparison figures (MAE, trial-by-trial, average activation).

3. **`pyedm_figures.ipynb`**
   Reproduces nonlinear forecasting results (Simplex ρ, predicted vs observed traces).

---

## Citation

If you use this repository, please cite:

```
Leonardis, E., Nagamori, A., Yang, Y., Park, J., Saunders, H.,  Azim, E., Pereira, T. D. (Accepted) Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics.  NeurIPS 2025: Data on the Brain & Mind Concrete Applications of AI to Neuroscience and Cognitive Science Workshop, San Diego, CA
```

---


[![Demo Video](https://img.youtube.com/vi/t2uQnJqCkKQ/0.jpg)](https://youtu.be/t2uQnJqCkKQ)
