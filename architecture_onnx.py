import onnx
import os

import netron
print(onnx.__version__)  # Doit afficher la version (ex: 1.14.0)
print(netron.__file__)  # Affiche le chemin du module (minuscule !)

import netron

# Chemin vers votre fichier ONNX
onnx_model_path = "decoder.onnx"  # Adaptez le nom

# Lance Netron et ouvre automatiquement le navigateur
netron.start(onnx_model_path)

# Attendre la fermeture (optionnel)
print("Visualisation lancée dans le navigateur. Appuyez sur Ctrl+C pour arrêter.")
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Fermeture de Netron")