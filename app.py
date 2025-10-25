import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import math


CLASS_NAMES = ['COVID', 'Normal', 'Viral Pneumonia'] 
SOFTMAX_THRESHOLD = 0.85  
ENTROPY_THRESHOLD = 1.0  


@st.cache_resource
def load_model():
    num_classes = len(CLASS_NAMES)
    model = models.vgg16(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    state_dict = torch.load("Covid19 Xray CNN Model Final.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def softmax_entropy(probs):
    """Calculate entropy of softmax probabilities"""
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return entropy


st.set_page_config(page_title="COVID X-Ray Classifier", page_icon="🧠", layout="centered")
st.title("🧠 COVID X-Ray Image Classification")
st.write("Upload a chest X-ray to classify it as COVID, Normal, or Viral Pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)


    img_tensor = transform(image).unsqueeze(0)


    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        max_prob, predicted_idx = torch.max(probs, 0)
        entropy = softmax_entropy(probs)

    if max_prob.item() < SOFTMAX_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        st.warning("⚠️ This image may not belong to trained classes (COVID/Normal/Viral).")
    else:
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        st.success(f"✅ Prediction: **{predicted_class}**")
        st.info(f"📊 Confidence: {max_prob.item():.2f} | Entropy: {entropy:.2f}")
