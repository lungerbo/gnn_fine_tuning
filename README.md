# HydraGNN GFM Fine-Tuning

Tools for fine-tuning a **single HydraGNN Graph Foundation Model (GFM)**  
(e.g., `gfm_0.229.pk`) on datasets such as **QM9**, **MD17**, and **Transition1x**.

---

## Overview
- Full and frozen-backbone fine-tuning modes  
- Configurable heads and tasks  
- Data preprocessing + evaluation (MAE, RMSE, R², UMAP, noise tests)

---

## Getting Started
- **HydraGNN**: https://github.com/ORNL/HydraGNN  
- **Pretrained GFM models**: https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024  
After installation, place your chosen checkpoint (e.g., `gfm_0.229.pk`) in the `checkpoints/` directory.

