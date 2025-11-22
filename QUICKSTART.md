# Quick Start Guide - SDXL Model Merger

## ⚡ Quick Setup (5 minutes)

### 1. Create Models Folder
```powershell
New-Item -ItemType Directory -Path "models" -Force
```

### 2. Download Models
Download and save to `models/` folder:
- juggernautXL_v8.safetensors
- realvisxlV40.safetensors  
- zavy_mix_xl.safetensors

### 3. Install Dependencies
```powershell
pip install torch safetensors
```

**For GPU (faster):**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121; pip install safetensors
```

### 4. Run Merge
```powershell
python merge.py
```

---

## 🎨 Customize Weights

Open `merge.py` and find this section (line ~155):

```python
MODELS_TO_MERGE = {
    "juggernaut": ("juggernautXL_v8.safetensors", 0.55),  # ← Change these
    "realvis": ("realvisxlV40.safetensors", 0.25),
    "zavymix": ("zavy_mix_xl.safetensors", 0.20),
}
```

**Note:** Weights must sum to ~1.0

---

## 📊 Output

Merged model saved to: `models/merged_sdxl.safetensors`

Use it in:
- Stable Diffusion WebUI
- Comfy UI
- InvokeAI

---

## ❓ Common Issues

| Issue | Solution |
|-------|----------|
| "No module safetensors" | `pip install safetensors --upgrade` |
| "Model not found" | Check filenames in models/ folder |
| Slow processing | Ensure GPU is being used (check terminal output) |

---

See `README.md` for detailed setup and troubleshooting.
