# decoder.py (corrigé)
import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Paramètres
DECODER_MODEL_PATH = "decoder.onnx"
LATENT_CSV_PATH = "latent_vector.csv"
LATENT_SHAPE_PATH = "latent_shape.npy"

def load_latent(csv_path, shape_path):
    latent_flat = np.loadtxt(csv_path, delimiter=",").astype(np.float32)
    latent_shape = np.load(shape_path)
    latent_tensor = latent_flat.reshape(latent_shape)
    return latent_tensor

def decode_latent(model_path, latent_tensor):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: latent_tensor})
    reconstructed_image = output[0].squeeze()  # Shape: (C, H, W)
    reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))  # CHW -> HWC
    
    # Inverser la normalisation PyTorch : x = 0.5 * (x + 1)
    reconstructed_image = 0.5 * (reconstructed_image + 1.0)
    reconstructed_image = np.clip(reconstructed_image * 255.0, 0, 255).astype(np.uint8)
    return reconstructed_image

def display_image(image_np):
    image = Image.fromarray(image_np)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Image reconstruite")
    plt.show()


latent_tensor = load_latent(LATENT_CSV_PATH, LATENT_SHAPE_PATH)
reconstructed_image = decode_latent(DECODER_MODEL_PATH, latent_tensor)
display_image(reconstructed_image)
