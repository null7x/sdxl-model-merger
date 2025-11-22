# SDXL Model Merger - Setup & Usage Guide

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- Approximately 100+ GB of free disk space

## 🗂️ Folder Structure Setup

Create the following directory structure in your project:

```
asaas/
├── merge.py
├── models/
│   ├── juggernautXL_v8.safetensors
│   ├── realvisxlV40.safetensors
│   └── zavy_mix_xl.safetensors
└── requirements.txt
```

### Step 1: Create Models Folder

1. In VS Code, open the terminal: **Ctrl + `** (backtick)
2. Run this command:

```powershell
New-Item -ItemType Directory -Path "models" -Force
```

### Step 2: Download & Place Model Files

1. Download the three models:
   - **Juggernaut XL v8**: Download `juggernautXL_v8.safetensors`
   - **RealVisXL V4**: Download `realvisxlV40.safetensors`
   - **ZavyMix XL**: Download `zavy_mix_xl.safetensors`

2. Place all `.safetensors` files in the `models/` folder you created

3. Verify the files are there by running:

```powershell
Get-ChildItem -Path "models" -Filter "*.safetensors"
```

You should see all three models listed.

## 📦 Install Requirements

### Create requirements.txt

The requirements.txt file should contain:

```
torch>=2.0.0
safetensors>=0.4.0
```

### Install Dependencies

Run in the VS Code terminal:

**For GPU (CUDA 12.1):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; pip install safetensors
```

**For GPU (CUDA 11.8):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; pip install safetensors
```

**For CPU only:**
```powershell
pip install torch safetensors
```

**Verify installation:**
```powershell
python -c "import torch; import safetensors; print('✓ All packages installed')"
```

## ▶️ How to Run in VS Code

### Method 1: Terminal (Recommended)

1. Open terminal: **Ctrl + `**
2. Ensure you're in the correct directory (should show `asaas` in path)
3. Run the script:

```powershell
python merge.py
```

### Method 2: Run Button

1. Open `merge.py` in the editor
2. Click the ▶️ **Run** button (top right)
3. Output appears in terminal

### Method 3: F5 (Debug)

1. Press **F5** to start debugging
2. Select "Python" as the environment
3. Monitor progress in Debug Console

## ⚙️ Adjusting Weights

To change the contribution of each model:

1. Open `merge.py` in editor
2. Find the section marked `# ============ ADJUSTABLE WEIGHTS ============` (around line 150)
3. Modify the weights:

```python
MODELS_TO_MERGE = {
    "juggernaut": ("juggernautXL_v8.safetensors", 0.55),  # Change 0.55
    "realvis": ("realvisxlV40.safetensors", 0.25),        # Change 0.25
    "zavymix": ("zavy_mix_xl.safetensors", 0.20),         # Change 0.20
}
```

**Important:** Weights should sum to approximately 1.0 (like 0.55 + 0.25 + 0.20 = 1.0)

Examples of other weight configurations:
- Equal merge: `0.33, 0.33, 0.34`
- Focus on Juggernaut: `0.70, 0.15, 0.15`
- Focus on RealVis: `0.30, 0.50, 0.20`

## 📊 Expected Output

When running successfully, you'll see:

```
============================================================
Stable Diffusion XL Model Merger
============================================================

Using device: cuda

Model Information:
------------------------------------------------------------
juggernaut: 55% weight
realvis: 25% weight
zavymix: 20% weight

============================================================
[1/3] Loading juggernaut...
Loading model: juggernautXL_v8.safetensors...
  ✓ Loaded 1000+ keys
...
[3/3] Saving merged model to merged_sdxl.safetensors...
  ✓ Successfully saved to models/merged_sdxl.safetensors
  File size: XX.XX GB

============================================================
✓ Merge completed successfully!
============================================================
```

## ⏱️ Timing Expectations

- **First run:** 15-30 minutes (includes model loading & processing)
- **Subsequent runs:** Similar timing (models are not cached)
- **GPU merge:** 5-15 minutes
- **CPU merge:** 30-60 minutes

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'safetensors'"

**Solution:** Install safetensors
```powershell
pip install safetensors --upgrade
```

### "FileNotFoundError: Model not found"

**Solution:** Check that:
1. Models are in `models/` folder
2. Filenames match exactly (case-sensitive)
3. Run: `Get-ChildItem models` to verify

### "CUDA out of memory"

**Solution:** Use CPU instead by editing merge.py:
```python
self.device = "cpu"  # Force CPU
```

### Script runs very slowly

**Solution:** Check which device is being used:
- Look for "Using device: cuda" (good) or "Using device: cpu" (slow)
- For GPU, install correct CUDA toolkit matching your GPU

## 📁 Output Location

The merged model is saved to:
```
asaas/models/merged_sdxl.safetensors
```

You can then use this file with:
- Stable Diffusion WebUI
- Comfy UI
- InvokeAI
- Other SD tools that support SDXL

## 📝 Script Features

- ✅ Loads all 3 models with progress tracking
- ✅ Weighted sum merging algorithm
- ✅ Shape mismatch detection and safe skipping
- ✅ Progress reporting every 100 keys
- ✅ GPU acceleration with CUDA fallback
- ✅ Detailed error handling
- ✅ Output file size reporting

## 💡 Tips

1. **Backup originals:** Keep the original model files as backup
2. **Test weights:** Try different weight combinations
3. **Monitor RAM:** Watch Task Manager while merging
4. **GPU:** Ensure GPU drivers are up to date for best performance

---

**Created:** 2025  
**Compatible with:** Stable Diffusion XL (SDXL)
