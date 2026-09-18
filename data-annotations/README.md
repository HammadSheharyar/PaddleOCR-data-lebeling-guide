# PaddleOCR / PPOCRLabel BIB Annotation Guide (Windows, CPU)

A practical setup and troubleshooting guide for labeling **cropped race BIB images** with **PPOCRLabel**, using PaddleOCR detection and Korean/English text recognition.

> This README is based on the setup and errors encountered during the actual Windows installation. The target machine has an older NVIDIA GPU that is not usable with the current PaddlePaddle GPU package, so this workflow intentionally uses **CPU inference**.

---

## 1. Goal

The annotation workflow is:

```text
Cropped BIB images
       |
       v
PPOCRLabel + PaddleOCR
       |
       +--> Text Detection: draw/detect text bounding boxes
       |
       +--> Text Recognition: recognize BIB number + Korean/English name
       |
       v
Manual cleanup
       |
       +--> Delete unrelated/incorrect text boxes
       +--> Keep only the BIB number and athlete name
       |
       v
Save annotations
       |
       +--> Detection labels
       +--> Recognition crops + recognition text labels
```

The important manual step is to remove extra OCR detections and keep only the required **BIB number** and **Korean/English name**. The detection labels can later be used for text-detection training, while recognition crops and text labels can be used for recognition-model training.

---

## 2. Tested Environment

This setup used:

- Windows
- Conda / Miniconda
- Python 3.11
- PaddlePaddle 3.3.1 (CPU)
- PaddleOCR 3.7.0
- PaddleX 3.7.2
- PPOCRLabel
- CUDA Toolkit 12.6 installed on the PC, but Paddle runs on CPU in this environment

The original environment name was:

```powershell
yolo_gemma
```

You can use a cleaner dedicated environment name such as `paddle_label_cpu`.

---

## 3. Create a Clean Conda Environment

A separate environment is strongly recommended because PPOCRLabel, PaddleOCR, PaddleX, PyQt, ModelScope, Torch, and other packages can conflict with packages from unrelated projects.

```powershell
conda create -n paddle_label_cpu python=3.11 -y
conda activate paddle_label_cpu
```

Verify:

```powershell
python --version
where python
pip --version
```

The Python and pip paths should point to the same Conda environment.

---

## 4. Install PaddlePaddle

### CPU installation

For the machine used in this project, the GPU path was abandoned because the installed NVIDIA GPU is too old for the required PaddlePaddle GPU build.

Install CPU PaddlePaddle:

```powershell
pip install paddlepaddle==3.3.1
```

If you already downloaded the Windows wheel locally:

```powershell
pip install .\paddlepaddle-3.3.1-cp311-cp311-win_amd64.whl
```

Verify the installation:

```powershell
python -c "import paddle; print(paddle.__version__)"
```

Expected version:

```text
3.3.1
```

Run Paddle's built-in check:

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

A successful CPU setup should end with output similar to:

```text
PaddlePaddle works well on 1 CPU.
PaddlePaddle is installed successfully!
```

Check the active device:

```powershell
python -c "import paddle; print('Version:', paddle.__version__); print('CUDA build:', paddle.device.is_compiled_with_cuda()); print('Device:', paddle.device.get_device())"
```

For this CPU setup, the important result is:

```text
CUDA build: False
Device: cpu
```

### Important: CUDA installed does NOT mean Paddle is using CUDA

The machine can still show CUDA Toolkit 12.6:

```powershell
nvcc --version
```

while Paddle reports:

```text
False
cpu
```

These are not contradictory. `nvcc` only proves that the CUDA toolkit is installed. Paddle must also have a compatible GPU build and the GPU itself must satisfy the supported compute capability.

---

## 5. Install PaddleOCR

```powershell
pip install paddleocr
```

In the recorded setup this installed:

- `paddleocr==3.7.0`
- `paddlex==3.7.2`

Check them:

```powershell
python -c "import paddleocr; print('PaddleOCR import OK')"
python -c "import paddlex; print('PaddleX:', paddlex.__version__)"
```

If OCR pipeline dependencies are missing, install the OCR extras explicitly:

```powershell
pip install "paddlex[ocr]==3.7.2"
```

Then verify again:

```powershell
python -c "import paddle; paddle.utils.run_check()"
python -c "import paddlex; print(paddlex.__version__)"
```

---

## 6. Install PPOCRLabel

Install the GUI annotation application:

```powershell
pip install PPOCRLabel
```

Check whether the command is available:

```powershell
PPOCRLabel --help
```

You can also check the installed package:

```powershell
pip show PPOCRLabel
```

---

# 7. How to Open the PPOCRLabel GUI

Always activate the environment first:

```powershell
conda activate paddle_label_cpu
```

If you kept the original environment:

```powershell
conda activate yolo_gemma
```

### Method A — normal launcher

Try this first:

```powershell
PPOCRLabel
```

### Method B — launch through Python

If the executable is not found or does not start correctly:

```powershell
python -c "from PPOCRLabel.PPOCRLabel import main; main()"
```

### Method C — launch with the required OCR models

For this BIB project:

```powershell
python -c "from PPOCRLabel.PPOCRLabel import main; main()" --lang en --det_model_name PP-OCRv5_mobile_det --rec_model_name korean_PP-OCRv5_mobile_rec
```

This configuration is intended to use:

- Detection: `PP-OCRv5_mobile_det`
- Recognition: `korean_PP-OCRv5_mobile_rec`

If PowerShell/Qt has a plugin-path problem, set the Qt plugin directory before launching:

```powershell
$env:QT_QPA_PLATFORM_PLUGIN_PATH="$env:CONDA_PREFIX\Lib\site-packages\PyQt5\Qt5\plugins"
```

Then run the GUI again:

```powershell
PPOCRLabel
```

or the Python launcher above.

Using `$env:CONDA_PREFIX` is better than hard-coding a username or Conda installation path.

---

# 8. Opening the Image Dataset in the GUI

After PPOCRLabel opens:

1. Open the directory containing the cropped BIB images.
2. Select an image.
3. Run automatic OCR/detection if required.
4. Inspect every detected text bounding box.
5. Delete boxes for sponsor text, logos, event text, noise, or unrelated text.
6. Keep the bounding box for the **BIB number**.
7. Keep the bounding box for the **Korean/English athlete name**.
8. Correct OCR text manually when recognition is wrong.
9. Save the annotation.
10. Move to the next image.

Do not blindly trust auto-labeling. The final annotations should contain only the text classes/content required by the BIB OCR project.

---

# 9. Expected Annotation Output

The project requires two related outputs.

### Detection data

The original image is retained with text bounding-box annotations. These annotations are used to train or fine-tune the text detection model.

Conceptually:

```text
image_001.jpg
    +-- BIB-number bounding box
    +-- athlete-name bounding box
```

### Recognition data

The selected text regions are cropped and paired with their correct transcription.

Conceptually:

```text
rec/
├── image_001_crop_0.jpg   -> 5090
└── image_001_crop_1.jpg   -> 김동현
```

The exact filenames and label files are generated by PPOCRLabel according to its export/save behavior.

---

# 10. Recommended Annotation Rules

For this dataset, keep the labels consistent.

```text
KEEP:
✓ BIB/race number
✓ Korean athlete name
✓ English athlete name when it is part of the required target

REMOVE:
✗ sponsor names
✗ event branding
✗ logos
✗ small unrelated numbers
✗ clothing text
✗ background signs
✗ duplicate/incorrect OCR boxes
✗ text belonging to another target when the crop is intended for one athlete
```

Before saving, check that the box tightly covers the text and that the transcription exactly matches the visible text.

---

# 11. Problems Encountered and Fixes

## Issue 1 — Paddle GPU build is unsuitable for the old GPU

### Symptom

CUDA 12.6 is installed, but the intended Paddle GPU setup cannot be used reliably on the machine.

### Fix

Use the CPU build:

```powershell
pip uninstall paddlepaddle-gpu -y
pip install paddlepaddle==3.3.1
```

Confirm:

```powershell
python -c "import paddle; print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())"
```

Expected:

```text
False
cpu
```

For labeling, CPU inference is acceptable even though it is slower.

---

## Issue 2 — `ccache` warning

Example:

```text
UserWarning: No ccache found.
```

### Meaning

This is normally a warning, not a Paddle installation failure.

### Action

If `paddle.utils.run_check()` finishes successfully, continue. `ccache` is mainly relevant when native source files need recompilation.

---

## Issue 3 — PIR warning about `place`

Example:

```text
Tensor do not have 'place' interface for pir graph mode
```

### Action

If the final Paddle check says that PaddlePaddle works successfully, this warning can generally be ignored for this setup.

---

## Issue 4 — `WinError 127` loading `torch\lib\shm.dll`

Observed error:

```text
OSError: [WinError 127] The specified procedure could not be found.
Error loading "...site-packages\torch\lib\shm.dll" or one of its dependencies.
```

### Likely cause

This points to a Torch/native DLL dependency problem inside the environment. It can occur when the environment contains incompatible Torch/CUDA/native-library versions.

### Best fix

Do not mix this annotation setup with an existing YOLO/Gemma/Torch environment unless necessary. Create a clean CPU labeling environment:

```powershell
conda create -n paddle_label_cpu python=3.11 -y
conda activate paddle_label_cpu
pip install paddlepaddle==3.3.1
pip install paddleocr
pip install PPOCRLabel
```

Then retry PPOCRLabel before adding unrelated PyTorch packages.

If Torch is not needed by your annotation workflow but was inherited from another project, the clean environment avoids the DLL conflict rather than patching around it.

---

## Issue 5 — ModelScope import crashes PPOCRLabel/PaddleX

The recorded traceback entered:

```text
modelscope
modelscope.utils.import_utils
modelscope.utils.ast_utils
modelscope.utils.file_utils
```

### Temporary workaround used during debugging

The PaddleX file was located with:

```powershell
python -c "import paddlex, os; print(os.path.join(os.path.dirname(paddlex.__file__), 'inference', 'utils', 'official_models.py'))"
```

Example location:

```text
C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddlex\inference\utils\official_models.py
```

It can be opened in VS Code:

```powershell
code "C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddlex\inference\utils\official_models.py"
```

The debugging change was:

```python
# Before
import modelscope

# Temporary defensive import
ms_hub_errors = None

try:
    import modelscope
except Exception:
    modelscope = None
```

### Important

Editing `site-packages` should be treated as a **last-resort workaround**, not the normal installation procedure. A package reinstall/update can overwrite the change.

Prefer first:

```powershell
pip install "paddlex[ocr]==3.7.2"
```

and verify whether `modelscope` imports correctly:

```powershell
python -c "import modelscope; print('ModelScope import OK')"
```

If the clean environment works, do not patch PaddleX source code.

---

## Issue 6 — PaddleX pipeline dependency error

Observed error:

```text
RuntimeError: A dependency error occurred during pipeline creation.
Please refer to the installation documentation to ensure all required dependencies are installed.
```

### Fix used

The base `paddlex` package was present:

```powershell
python -c "import paddlex; print(paddlex.__version__)"
```

which returned:

```text
3.7.2
```

The OCR-specific extras were then installed:

```powershell
pip install "paddlex[ocr]==3.7.2"
```

After installation, close and reopen the terminal, reactivate the environment, and retry the GUI.

---

## Issue 7 — Qt platform plugin / GUI does not open

Set the Qt plugin directory using the active Conda environment:

```powershell
$env:QT_QPA_PLATFORM_PLUGIN_PATH="$env:CONDA_PREFIX\Lib\site-packages\PyQt5\Qt5\plugins"
```

Check that it exists:

```powershell
Test-Path $env:QT_QPA_PLATFORM_PLUGIN_PATH
```

Then launch:

```powershell
PPOCRLabel
```

If the variable causes problems in a later session, remove it:

```powershell
Remove-Item Env:QT_QPA_PLATFORM_PLUGIN_PATH
```

and reopen the terminal.

---

## Issue 8 — `PPOCRLabel` command is not recognized

First make sure the environment is active:

```powershell
conda activate paddle_label_cpu
```

Check:

```powershell
pip show PPOCRLabel
where python
where PPOCRLabel
```

If the launcher still cannot be found, use:

```powershell
python -c "from PPOCRLabel.PPOCRLabel import main; main()"
```

---

# 12. Quick Start — Every Time You Want to Label Images

Once installation is complete, you should **not reinstall everything every time**.

Open **Anaconda Prompt** or **PowerShell**:

```powershell
conda activate paddle_label_cpu
cd F:\paddleocr-finetuned\data-annotations\dataset-2026
$env:QT_QPA_PLATFORM_PLUGIN_PATH="$env:CONDA_PREFIX\Lib\site-packages\PyQt5\Qt5\plugins"
PPOCRLabel
```

If you use the original environment:

```powershell
conda activate yolo_gemma
cd F:\paddleocr-finetuned\data-annotations\dataset-2026
$env:QT_QPA_PLATFORM_PLUGIN_PATH="$env:CONDA_PREFIX\Lib\site-packages\PyQt5\Qt5\plugins"
python -c "from PPOCRLabel.PPOCRLabel import main; main()" --lang en --det_model_name PP-OCRv5_mobile_det --rec_model_name korean_PP-OCRv5_mobile_rec
```

That is the main **daily GUI startup procedure**.

---

# 13. One-Time Installation vs Daily Use

| Task | One time | Every session |
|---|:---:|:---:|
| Create Conda environment | ✓ | |
| Install PaddlePaddle | ✓ | |
| Install PaddleOCR/PaddleX | ✓ | |
| Install PPOCRLabel | ✓ | |
| Activate Conda environment | | ✓ |
| `cd` to project/dataset | | ✓ |
| Set Qt plugin path if required | | ✓ |
| Open PPOCRLabel | | ✓ |
| Open image directory | | ✓ |
| Clean/correct annotations | | ✓ |
| Save labels | | ✓ |

---

# 14. Installation Verification Checklist

Run these commands before debugging the GUI itself:

```powershell
python --version
python -c "import paddle; print('Paddle:', paddle.__version__, 'Device:', paddle.device.get_device())"
python -c "import paddleocr; print('PaddleOCR OK')"
python -c "import paddlex; print('PaddleX:', paddlex.__version__)"
python -c "import modelscope; print('ModelScope OK')"
python -c "import PyQt5; print('PyQt5 OK')"
pip show PPOCRLabel
```

If one command fails, fix that dependency before trying to launch the entire GUI. This makes troubleshooting much easier.

---

# 15. Recommended Project Layout

A clean dataset structure helps prevent accidental overwriting:

```text
data-annotations/
├── README.md
├── raw_crops/
│   ├── event_01/
│   └── event_02/
├── annotation_work/
│   └── dataset-2026/
├── exports/
│   ├── detection/
│   └── recognition/
└── backups/
```

Keep original images read-only or backed up. Do annotation work on a copied working dataset.

---

# 16. Recommended Backup Practice

Annotation work is expensive to recreate. After every meaningful labeling session, back up:

- original/current images,
- detection annotation text files,
- recognition label files,
- generated recognition crops,
- any class/dictionary/configuration files required for training.

A simple dated backup naming convention is useful:

```text
backups/
├── 2026-09-18/
├── 2026-09-19/
└── ...
```

---

# 17. Clean Reinstallation Procedure

If dependency conflicts become too complicated, rebuilding the environment is usually safer than repeatedly modifying `site-packages`.

```powershell
conda deactivate
conda remove -n paddle_label_cpu --all -y

conda create -n paddle_label_cpu python=3.11 -y
conda activate paddle_label_cpu

python -m pip install --upgrade pip
pip install paddlepaddle==3.3.1
pip install paddleocr
pip install "paddlex[ocr]==3.7.2"
pip install PPOCRLabel
```

Verify Paddle:

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

Then launch PPOCRLabel.

---

# 18. Final Workflow Summary

```text
ONE-TIME SETUP
Conda env
   |
   v
PaddlePaddle CPU
   |
   v
PaddleOCR + PaddleX OCR dependencies
   |
   v
PPOCRLabel
   |
   v
Verify imports

EVERY LABELING SESSION
Activate environment
   |
   v
Open project directory
   |
   v
Set Qt plugin path (only if needed)
   |
   v
Launch PPOCRLabel GUI
   |
   v
Open cropped BIB images
   |
   v
Auto-detect / recognize text
   |
   v
Delete unwanted boxes
   |
   v
Keep BIB number + athlete name
   |
   v
Correct transcription
   |
   v
Save detection + recognition annotations
   |
   v
Back up annotation files
```

---

## Notes

This README intentionally documents both the successful setup and the failures encountered during installation. In particular, it preserves the CPU fallback, the Torch `shm.dll` failure, the ModelScope/PaddleX import issue, the Qt plugin-path workaround, and the missing PaddleX OCR dependency problem so the environment can be reproduced or repaired later.
