import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
# import matplotlib.pyplot as plt

# Load the Pix2Pix GAN model (update this with your specific model path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming the model is a simple Pix2Pix model (you need to replace this with your actual model)
class Generator(torch.nn.Module):
    # Define the architecture of your generator (replace this with your own)
    def __init__(self):
        super(Generator, self).__init__()
        # Add layers for the Pix2Pix Generator (simplified for illustration)
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x  # Simplified, replace with actual model's forward pass

# Load the pre-trained model
model = Generator().to(device)
# model.load_state_dict(torch.load('pix2pix_model.pth', map_location=device))
model.load_state_dict(torch.load('pix2pix_model.pth', map_location=device), strict=False)
model.eval()

# Function to preprocess image for model input
def preprocess_image(image):
    # Convert image to RGB and resize to match model input
    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 or the expected input size of your model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to the correct device
    return image

# Function to generate colorized image
def generate_colorized_image(model, image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Perform inference using the Pix2Pix model
    with torch.no_grad():
        output = model(processed_image)
    
    # Convert output to a usable format for display (denormalize)
    output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Remove batch dimension and convert to HWC
    output = (output + 1) * 127.5  # Denormalize back to [0, 255]
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

# Streamlit app layout
st.title("Pix2Pix Image Colorization")
st.write("Upload a black-and-white image and see it colorized using the Pix2Pix GAN model.")

# Upload image
uploaded_image = st.file_uploader("Upload a black-and-white image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_image)
    
    # Display the original image
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Colorize the image using the Pix2Pix model
    colorized_image = generate_colorized_image(model, image)
    
    # Display the colorized image
    st.image(colorized_image, caption="Colorized Image", use_column_width=True)
