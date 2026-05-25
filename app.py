import streamlit as st
import os
import io
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# MUST BE FIRST
st.set_page_config(
    page_title="MedGemma AI | Clinical Analysis Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

from PIL import Image
import numpy as np
from fpdf import FPDF
from utils.image_processor import MedicalImageProcessor
from core.model_handler import MedGemmaHandler

# --- Load Custom CSS ---
if os.path.exists("assets/styles.css"):
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Initialize Session State ---
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
if "raw_hu" not in st.session_state:
    st.session_state.raw_hu = []
if "dicom_meta" not in st.session_state:
    st.session_state.dicom_meta = []
if "expert_sign_off" not in st.session_state:
    st.session_state.expert_sign_off = False

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
    st.title("Nexus Medical AI")
    st.markdown("---")
    
    st.subheader("🛠 Configuration")
    hf_token = st.text_input(
        "Hugging Face Token", 
        value=os.getenv("HF_TOKEN", ""),
        type="password", 
        help="Required to access MedGemma 1.5. Can also be set in .env file."
    )
    scan_type = st.selectbox("Scan Modality", ["CT Scan", "MRI Scan", "Mammography"])
    
    use_mock = st.checkbox("Use Demo Mode (Mock AI)", value=True, help="Toggle this if you don't have a GPU or HF weights locally.")
    
    st.markdown("---")
    st.info("MedGemma 1.5-4B-IT is optimized for medical visual reasoning.")

# --- Helper Functions ---
def _latin1(text):
    """Core PDF fonts only support latin-1; replace unsupported characters safely."""
    return str(text).encode("latin-1", "replace").decode("latin-1")

def generate_pdf(report, scan_type, expert_name):
    pdf = FPDF()
    pdf.add_page()

    # Disclaimer banner — every exported report is explicitly marked as AI-generated
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(198, 40, 40)
    pdf.cell(0, 8, txt="AI-GENERATED - REQUIRES CLINICAL REVIEW", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=_latin1(f"Clinical Analysis Report - {scan_type}"), ln=True, align='C')
    pdf.ln(4)

    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt=_latin1(f"AI Confidence Level: {report.get('confidence', 'N/A')}"), ln=True)
    pdf.ln(4)

    sections = [
        ("Findings", report.get("findings", "")),
        ("Impression", report.get("impression", "")),
        ("Recommendations", report.get("recommendations", "")),
    ]
    for title, body in sections:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=title, ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 7, txt=_latin1(body if body else "N/A"))
        pdf.ln(3)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, txt=_latin1(f"Digitally Validated By: {expert_name}"), ln=True)

    return bytes(pdf.output())

# --- Main App Logic ---
st.title("🩺 MedGemma Clinical Analysis")
st.write("Advance your clinical workflow with multimodal medical reasoning.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.subheader("📁 Scan Data Upload")
    
    # Accept any medical or standard image format
    supported_types = ["dcm", "nii", "nii.gz", "jpg", "jpeg", "png", "bmp", "webp", "tiff"]
    uploaded_files = st.file_uploader(
        f"Upload {scan_type} files", 
        type=supported_types, 
        accept_multiple_files=True
    )

    if uploaded_files:
        # processed_images feed the model; raw_hu and dicom_meta are kept in parallel
        # (None for non-DICOM entries) to power window/level and the metadata panel.
        st.session_state.processed_images = []
        st.session_state.raw_hu = []
        st.session_state.dicom_meta = []
        for f in uploaded_files:
            file_ext = f.name.lower()

            # 1. Handle DICOM
            if file_ext.endswith(".dcm"):
                hu_data, ds = MedicalImageProcessor.load_dicom_slice(io.BytesIO(f.read()))
                rgb_slice = MedicalImageProcessor.process_ct_rgb(hu_data)
                resized = MedicalImageProcessor.resize_for_model(rgb_slice)
                st.session_state.processed_images.append(Image.fromarray(resized))
                st.session_state.raw_hu.append(hu_data)
                st.session_state.dicom_meta.append(MedicalImageProcessor.extract_dicom_metadata(ds))

            # 2. Handle NIfTI
            elif file_ext.endswith(".nii") or file_ext.endswith(".nii.gz"):
                suffix = ".nii.gz" if file_ext.endswith(".nii.gz") else ".nii"
                slices = MedicalImageProcessor.extract_nifti_slices(f.read(), suffix=suffix)
                st.session_state.processed_images.extend(slices)
                st.session_state.raw_hu.extend([None] * len(slices))
                st.session_state.dicom_meta.extend([None] * len(slices))

            # 3. Handle Standard Image Formats
            else:
                processed_img = MedicalImageProcessor.prepare_any_image(f.read())
                if processed_img:
                    st.session_state.processed_images.append(processed_img)
                    st.session_state.raw_hu.append(None)
                    st.session_state.dicom_meta.append(None)

        # Truncate to a reasonable amount, keeping all parallel lists aligned
        limit = 5
        st.session_state.processed_images = st.session_state.processed_images[:limit]
        st.session_state.raw_hu = st.session_state.raw_hu[:limit]
        st.session_state.dicom_meta = st.session_state.dicom_meta[:limit]

        # Display Preview
        if st.session_state.processed_images:
            count = len(st.session_state.processed_images)
            slice_idx = st.slider("Select Slice/Image to View", 0, count - 1, 0) if count > 1 else 0

            raw = st.session_state.raw_hu[slice_idx]
            if raw is not None:
                # Window/Level controls — meaningful for CT (Hounsfield Unit) data
                wl1, wl2 = st.columns(2)
                with wl1:
                    wc = st.slider("Window Center (Level)", -1000, 1000, 40)
                with wl2:
                    ww = st.slider("Window Width", 1, 2000, 400)
                windowed = MedicalImageProcessor.apply_window(raw, wc, ww)
                st.image(windowed, caption=f"CT View {slice_idx+1} — WC {wc} / WW {ww}", use_container_width=True)

                meta = st.session_state.dicom_meta[slice_idx]
                if meta:
                    with st.expander("DICOM Study Metadata (anonymized)"):
                        for k, v in meta.items():
                            st.text(f"{k}: {v}")
            else:
                st.image(st.session_state.processed_images[slice_idx], caption=f"Scan Preview {slice_idx+1}", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Analysis Preview")
    
    analyze_btn = st.button("GENERATE CLINICAL INSIGHTS", disabled=not st.session_state.processed_images)
    
    if analyze_btn:
        with st.spinner("Analyzing scan data..."):
            if use_mock:
                handler = MedGemmaHandler()
                raw_report = handler.mock_analyze(scan_type)
                st.session_state.report_data = MedGemmaHandler.parse_report(raw_report)
            else:
                if not hf_token:
                    st.error("Please enter a Hugging Face Token in the sidebar.")
                else:
                    try:
                        handler = MedGemmaHandler()
                        handler.initialize(hf_token=hf_token)
                        raw_report = handler.analyze(st.session_state.processed_images)
                        st.session_state.report_data = MedGemmaHandler.parse_report(raw_report)
                    except Exception as e:
                        msg = str(e).lower()
                        if any(k in msg for k in ["couldn't connect", "connection", "offline", "timed out", "max retries"]):
                            st.error(
                                "Couldn't reach Hugging Face to download the model. Check your internet "
                                "connection and try again, or enable Demo Mode in the sidebar."
                            )
                        elif any(k in msg for k in ["gated", "401", "403", "authoriz", "access to", "awaiting"]):
                            st.error(
                                "Access to MedGemma was denied. Make sure you've accepted the model license on its "
                                "Hugging Face page and that your token is valid, then try again."
                            )
                        elif any(k in msg for k in ["404", "not found", "does not exist"]):
                            st.error("The model repository could not be found. Verify the model ID is correct.")
                        else:
                            st.error("Model initialization failed. See details below, or enable Demo Mode.")
                        with st.expander("Technical details"):
                            st.code(str(e))

    if st.session_state.report_data:
        report = st.session_state.report_data

        # Confidence badge
        conf = report.get("confidence", "N/A")
        conf_color = {"High": "#2e7d32", "Moderate": "#f9a825", "Low": "#c62828"}.get(conf, "#607d8b")
        st.markdown(
            f"<span style='background:{conf_color};color:white;padding:4px 12px;"
            f"border-radius:12px;font-size:0.85rem;font-weight:600;'>AI Confidence: {conf}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("### Structured Findings")
        st.caption("Review and edit each section before sign-off.")
        report["findings"] = st.text_area("Findings", value=report.get("findings", ""), height=160)
        report["impression"] = st.text_area("Impression", value=report.get("impression", ""), height=100)
        report["recommendations"] = st.text_area("Recommendations", value=report.get("recommendations", ""), height=100)

        st.markdown("---")
        st.subheader("✍️ Validation & Signature")
        expert_name = st.text_input("Full Name of Reviewing Clinician")
        st.session_state.expert_sign_off = st.checkbox("Sign-off: I confirm these findings are accurate.")

        if st.session_state.expert_sign_off and expert_name:
            pdf_bytes = generate_pdf(report, scan_type, expert_name)
            st.download_button(
                label="📥 DOWNLOAD FINAL REPORT (PDF)",
                data=pdf_bytes,
                file_name=f"Clinical_Final_{scan_type.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Complete the sign-off to generate the final PDF report.")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("AI-assisted diagnostics should be interpreted by qualified healthcare professionals. Based on Google MedGemma 1.5.")
