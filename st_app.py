import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Set page configuration
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered"
)

# ---- 1. Model and Processor Setup ----
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    """
    Loads the CLIP model and processor directly from Hugging Face.
    This will download the model on the first run.
    """
    model_name = "openai/clip-vit-base-patch32"
    st.info(f"Loading model '{model_name}' from Hugging Face Hub... This may take a moment on the first run.")
    
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        st.success("Model loaded successfully!")
        return model, processor, device
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.error("Please check your internet connection and ensure you have the 'transformers' library installed.")
        st.stop()

# Load the model, processor, and device
model, processor, device = load_model()

# ---- 2. Define Prompt Groups ----
document_prompts = [
    "a photo of an Aadhaar card", "a photo of a driving licence", 
    "a photo of a vehicle registration certificate", "a photo of a metal plate with chassis number",
    "a photo of a metal plate with engine number", "a photo of an identity card",
    "a photo of a printed document",
]

nondocument_prompts = [
    "a photo of a damaged car part", "a photo of a car accident", "a photo of a vehicle",
    "a photo of a car interior", "a photo of the road", "photo of motocylce", "photo of bike",
    "photo of two wheeler", "photo of three wheeler", "speedometer of vehicles of any kind",
    "any type of vehicle part part", "photo of car engine with details engraved on a metal plate",
    "photo of any vehicle with number plate visible", "any text written on any metal plate or part",
    "any number or text engraved on metal part of car", "any sticker pasted on windshield of any vehilce",
    "any sticker on glass in close up or zoom out on the vehicle",
    "engraved text on metal part of engine area in vehicle even with corrossion",
    "number plate of any vehicle in closeup even if partially visible",
    "Any text engraved or printed on any shiny surface like marble or glass"
]


# ---- 3. Image Classification Function ----
def classify_image(image: Image.Image):
    """
    Classifies a single image and returns its label ('document' or 'non-document')
    and the associated confidence score.
    """
    all_prompts = document_prompts + nondocument_prompts
    
    inputs = processor(
        text=all_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # Group scores by taking the maximum probability within each category
    doc_score = max(probs[:len(document_prompts)])
    nondoc_score = max(probs[len(document_prompts):])
    
    if doc_score > nondoc_score:
        return "📄 Document", doc_score
    else:
        return "🖼️ Non-Document", nondoc_score

# ---- 4. Streamlit UI ----
st.title("📄 Document vs. Non-Document Classifier")
st.markdown("Upload an image, and the app will classify it as a **document** or a **non-document** using the OpenAI CLIP model.")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    # Open and display the image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Center the image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("") # Add some space

        # Classify the image when the button is clicked
        if st.button("Classify Image", use_container_width=True, type="primary"):
            with st.spinner("Analyzing the image..."):
                label, score = classify_image(image)
                
                if "Document" in label:
                    st.success(f"**Classification:** {label}")
                else:
                    st.info(f"**Classification:** {label}")
                    
                st.metric(label="Confidence Score", value=f"{score:.2%}")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.info("Please upload an image to begin classification.")
