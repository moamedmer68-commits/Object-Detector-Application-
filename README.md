#  Simple Object Detector using Hugging Face + Streamlit

A lightweight **object detection web app** built with **Streamlit** and **Hugging Face Transformers**.  
It uses Facebook’s **DETR (DEtection TRansformer)** model to detect and label objects within uploaded images.

---

##  Features
✅ Upload any image (`.jpg`, `.jpeg`, `.png`)  
✅ Automatically resizes the image to `255×255` for consistent detection  
✅ Uses **facebook/detr-resnet-50** from Hugging Face Transformers  
✅ Draws **red bounding boxes** and labels for detected objects  
✅ Displays **object names and confidence scores**  
✅ Fully interactive **Streamlit web interface**

---

##  Tech Stack
| Component | Description |
|------------|-------------|
| **Streamlit** | For the interactive web interface |
| **Transformers (Hugging Face)** | To load the pretrained DETR model |
| **PyTorch** | Deep learning backend used by DETR |
| **timm** | Image model utilities required by DETR |
| **Pillow (PIL)** | Image processing and drawing |
| **NumPy** | Array manipulation and conversion |

---

##  Installation

1️ Clone the repository or copy the code:
```bash
git clone https://github.com/YOUR_USERNAME/simple-object-detector.git
cd simple-object-detector
