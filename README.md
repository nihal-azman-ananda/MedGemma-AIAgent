# MedGemma Clinical Analysis Dashboard

A locally-hosted medical imaging analysis platform powered by **Google MedGemma 1.5-4B-IT** — a multimodal language model fine-tuned for medical visual reasoning. The application provides AI-assisted interpretation of DICOM, NIfTI, and standard image formats with a built-in Human-in-the-Loop (HITL) clinician validation workflow before any report is finalized.

> **Regulatory Notice:** This system is intended for research and clinical assistive use only. All final medical decisions must be made by a licensed healthcare professional.

---

## Features

- **Local inference** — runs entirely on your machine, no external API calls
- **Multi-format support** — DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`), and standard formats (JPG, PNG, TIFF, BMP, WebP)
- **Advanced CT preprocessing** — multi-window RGB encoding (soft tissue, bone, lung) for richer model input
- **4-bit quantization** — NF4 quantization via BitsAndBytes reduces VRAM requirement from ~8GB to ~3GB on CUDA
- **Human-in-the-Loop validation** — clinician must review, edit, and sign off before a report can be exported
- **PDF report export** — generates a signed clinical report with findings and clinician attribution
- **Demo mode** — fully functional UI without model weights, for testing and development

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Model | Google MedGemma 1.5-4B-IT |
| Inference | PyTorch + Hugging Face Transformers |
| Quantization | BitsAndBytes (NF4 4-bit) |
| DICOM processing | pydicom |
| NIfTI processing | nibabel |
| Image processing | OpenCV, Pillow |
| PDF generation | fpdf2 |

---

## Project Structure

```
MedGemma_AIAgent/
├── app.py                  # Main Streamlit application and UI logic
├── core/
│   └── model_handler.py    # MedGemma model loading and inference
├── utils/
│   └── image_processor.py  # Medical image preprocessing pipeline
├── assets/
│   └── styles.css          # Custom UI styling
├── requirements.txt
└── .env                    # (optional) HF token config — not committed
```

---

## Architecture

```
User Uploads Scan
       |
       v
Image Processor (utils/image_processor.py)
  - DICOM: pydicom → HU conversion → multi-window RGB encoding
  - NIfTI: nibabel → slice extraction → multi-window RGB encoding
  - Standard: Pillow → RGB conversion
  - All formats → resize to 896x896 (MedGemma input requirement)
       |
       v
MedGemma Handler (core/model_handler.py)
  - Singleton model instance (loaded once, reused across reruns)
  - 4-bit NF4 quantization on CUDA / full float32 on CPU
  - Multimodal chat template: [image(s) + text prompt] → generated findings
       |
       v
Streamlit UI (app.py)
  - Editable findings text area
  - Clinician name + sign-off checkbox
  - PDF export (only unlocked after sign-off)
```

---

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended — runs on CPU but will be slow)
- A Hugging Face account with access to the MedGemma model

### Requesting MedGemma Access

MedGemma is a gated model. Before running, you must:

1. Create a free account at huggingface.co
2. Navigate to the MedGemma model page (google/medgemma-1.5-4b-it) and accept Google's terms of use
3. Generate an access token from Settings > Access Tokens in your HF account

---

## Installation

**1. Clone the repository**
```bash
git clone <repo-url>
cd MedGemma_AIAgent
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install PyTorch with CUDA support**

Install torch first with the appropriate CUDA build before installing the rest of the requirements. This ensures pip does not pull the CPU-only version from PyPI.

```bash
# CUDA 12.8 (recommended — compatible with CUDA 13.x drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only (no GPU)
pip install torch
```

**4. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

---

## Configuration

To avoid entering your Hugging Face token in the UI on every run, create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token_here
```

The app loads this automatically on startup via python-dotenv.

---

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Usage

1. **Select scan modality** (CT, MRI, Mammography) from the sidebar
2. **Upload scan files** — supports DICOM, NIfTI, and standard image formats
3. **Toggle Demo Mode** in the sidebar if you want to test without model weights
4. Click **Generate Clinical Insights** to run inference
5. **Review and edit** the AI-generated findings in the text area
6. Enter your **name** and check the **sign-off box** to validate the report
7. Click **Download Final Report** to export the signed PDF

---

## GPU Memory Requirements

| Mode | Approximate VRAM |
|---|---|
| 4-bit quantized (CUDA) | ~3 GB |
| Full precision (CUDA) | ~8 GB |
| CPU (float32) | System RAM |

4-bit quantization is enabled automatically when a CUDA GPU is detected.
