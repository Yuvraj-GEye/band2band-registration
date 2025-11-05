# 🛰️ Band-to-Band Registration (BBR) on Simulated MSI Stacks

This repository provides a workflow to **simulate band misalignments** (translations and rotations) in multispectral imagery (MSI) and perform **band-to-band registration (BBR)** to correct them.  
The workflow also computes the **registration accuracy (RMSE)** over the aligned stacks.

---

## 📘 Overview

The project performs the following key steps:

1. **Simulation of Misalignments**
   - Create synthetic MSI stacks with:
     - **Test 1:** Translated bands
     - **Test 2:** Translated + Rotated bands

2. **Band-to-Band Registration (BBR)**
   - Aligns each band to a selected **reference band**.
   - Uses **SIFT-based feature extraction** and **GPU-optimized RANSAC** for transformation estimation.
   - Applies **Affine transformation** for geometric correction.
   - Exports the **registered MSI stacks as GeoTIFFs**.

3. **Accuracy Evaluation**
   - Computes **RMSE** between reference and registered band keypoints.

---

## ⚙️ Core Components

| File | Description |
|------|--------------|
| `notebooks/bbr_msi_registration.ipynb` | Main notebook for executing the workflow |
| `extract_sift_features.py` | Implements block-based SIFT feature extraction |
| `ransacgpu.py` | GPU-optimized RANSAC model for robust transformation fitting |
| `affine_utils.py` | Helper functions for affine matrix decomposition |
| `requirements.txt` | Python dependencies |

---

## 🧠 Dependencies

Make sure you have the required packages installed:

```bash
pip install -r requirements.txt
