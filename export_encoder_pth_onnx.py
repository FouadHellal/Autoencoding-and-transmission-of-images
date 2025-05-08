import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        channels = [48, 64, 128]
        bottleneck_dim = [32, 2]
        padding = 1

        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels[2], bottleneck_dim[0], kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(bottleneck_dim[0], bottleneck_dim[1], kernel_size=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.encoder(x)

def export_encoder(weights_file, onnx_file):
    # Configuration
    input_dim = 128
    
    # Création du dossier si inexistant
    #os.makedirs(onnx_dir, exist_ok=True)

    # Chargement du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder().to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()

    # Input exemple (batch_size=1, channels=3, height=128, width=128)
    dummy_input = torch.randn(1, 3, input_dim, input_dim).to(device)

    # Export ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )

    print(f"\nEncodeur ONNX exporté avec succès à: {onnx_file}")
    print(f"Shape d'entrée attendue: {dummy_input.shape}")
    print(f"Shape de sortie: {model(dummy_input).shape}")

weights_file = "encoder.pth"

onnx_file = "encoder.onnx"

export_encoder(weights_file, onnx_file)