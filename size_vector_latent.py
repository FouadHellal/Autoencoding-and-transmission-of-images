import onnx
import onnxruntime as ort

# Charger les modèles
encoder_model = onnx.load("encoder.onnx")
decoder_model = onnx.load("decoder.onnx")

# Méthode 1: Inspection via ONNX (sans exécution)
# Pour l'encodeur - trouver la sortie
print("Encoder output shape:")
for output in encoder_model.graph.output:
    print(f"{output.name}: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")

# Pour le décodeur - trouver l'entrée
print("\nDecoder input shape:")
for input in decoder_model.graph.input:
    print(f"{input.name}: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")