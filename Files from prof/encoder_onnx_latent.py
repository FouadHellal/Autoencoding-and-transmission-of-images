# encoder.py (corrigé)
import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd

# Paramètres
ENCODER_MODEL_PATH = "encoder.onnx"
IMAGE_PATH = "IMG_1189.jpg"
LATENT_CSV_PATH = "latent_vector.csv"
LATENT_SHAPE_PATH = "latent_shape.npy"
IMAGE_SIZE = (128, 128)

def preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = (image_np - 0.5) / 0.5
    image_np = np.transpose(image_np, (2, 0, 1))  # HWC -> CHW
    image_np = np.expand_dims(image_np, axis=0)   # Batch size = 1
    return image_np

def encode_image(model_path, image_tensor):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image_tensor})
    latent_tensor = output[0]  # Shape: (1, C, H, W)
    return latent_tensor

def save_latent(latent_tensor, csv_path, shape_path):
    latent_flat = latent_tensor.flatten()
    np.savetxt(csv_path, latent_flat, delimiter=",")
    np.save(shape_path, latent_tensor.shape)
    print(f"Vecteur latent sauvegardé dans {csv_path}")
    print(f"Shape sauvegardée dans {shape_path}")

image_tensor = preprocess_image(IMAGE_PATH, IMAGE_SIZE)
latent_tensor = encode_image(ENCODER_MODEL_PATH, image_tensor)
save_latent(latent_tensor, LATENT_CSV_PATH, LATENT_SHAPE_PATH)
