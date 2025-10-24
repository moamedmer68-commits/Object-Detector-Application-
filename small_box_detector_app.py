import os
import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont

os.system("pip install transformers")
os.system("pip install timm")
# ----------------------------------------------------
# Streamlit App Setup
# ----------------------------------------------------
st.set_page_config(page_title="Simple Object Detector")
st.title("Simple Object Detector")
st.write("Upload an image and detect objects in it.")


# ----------------------------------------------------
# Load Hugging Face Model
# ----------------------------------------------------
@st.cache_resource
def load_model():
    """
    Load the object detection model from Hugging Face.
    This model can detect various objects in an image.
    """
    st.info("Loading small Hugging Face model...")
    model = pipeline("object-detection", model="facebook/detr-resnet-50")
    st.success("Model loaded successfully!")
    return model

detector = load_model()

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def resize_image(image, size=(255, 255)):
    """
    Resize the input image to exactly 255x255 pixels.

    Args:
        image (PIL.Image): Input image.
        size (tuple): Desired output size (width, height).

    Returns:
        PIL.Image: Resized image.
    """
    return image.resize(size)

def draw_boxes(image, detections, threshold=0.4, thickness=5):
    """
    Draw thick red boxes and labels for each detected object.

    Args:
        image (PIL.Image): Image on which to draw.
        detections (list): List of detected objects from the model.
        threshold (float): Minimum confidence to display a detection.
        thickness (int): Thickness of the bounding box in pixels.

    Returns:
        PIL.Image: Annotated image with boxes and labels.
    """
    draw = ImageDraw.Draw(image)
    w, h = image.size
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for det in detections:
        if det["score"] < threshold:
            continue
        box = det["box"]
        label = det["label"]
        score = det["score"]

        left, top, right, bottom = (
            int(box["xmin"]),
            int(box["ymin"]),
            int(box["xmax"]),
            int(box["ymax"])
        )

        # Draw thick rectangle
        for offset in range(thickness):
            draw.rectangle(
                [left - offset, top - offset, right + offset, bottom + offset],
                outline=(255, 0, 0)
            )

        # Draw label background and text
        text = f"{label} ({score*100:.1f}%)"
        text_w = draw.textlength(text, font=font)
        text_h = 12
        draw.rectangle(
            [left, top - text_h - 2, left + text_w + 6, top],
            fill=(255, 0, 0)
        )
        draw.text((left + 3, top - text_h - 1), text, fill=(255, 255, 255), font=font)

    return image

# ----------------------------------------------------
# Streamlit User Interface
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize the image to 255x255
    resized = resize_image(image, (255, 255))
    st.image(resized, caption="Resized Input Image (255x255)", use_container_width=True)

    # Detect objects
    st.write("Detecting objects...")
    detections = detector(resized)

    # Draw boxes on the image
    annotated = resized.copy()
    annotated = draw_boxes(annotated, detections, threshold=0.4, thickness=5)
    st.image(annotated, caption="Detection Result", use_container_width=True)

    # Display detected objects and their confidence
    found = [
        f"- {d['label']} ({d['score']*100:.1f}% confidence)"
        for d in detections if d['score'] > 0.4
    ]
    if found:
        st.success("Objects detected:\n" + "\n".join(found))
    else:
        st.warning("No objects detected with sufficient confidence.")
