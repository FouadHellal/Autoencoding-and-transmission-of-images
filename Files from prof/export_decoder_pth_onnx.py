import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        dec_channels = [32, 48, 64, 128]
        bottleneck_dim = [32, 2]
        padding = 1

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_dim[1], bottleneck_dim[0], kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(bottleneck_dim[0], dec_channels[3], kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(dec_channels[3], dec_channels[2], kernel_size=3, padding=padding),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(dec_channels[2], dec_channels[1], kernel_size=3, padding=padding),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(dec_channels[1], dec_channels[0], kernel_size=3, padding=padding),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dec_channels[0], 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

def export_decoder(weights_file, onnx_fil):
    # Configuration
    
   # Chargement du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Decoder().to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()

    # Input exemple correspondant à la sortie de l'encodeur
    # (batch_size=1, channels=2, height=4, width=4)
    dummy_input = torch.randn(1, 2, 16, 16).to(device)

    # Export ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['latent_input'],
        output_names=['decoded_output'],
        dynamic_axes={
            'latent_input': {0: 'batch_size'},
            'decoded_output': {0: 'batch_size'}
        },
        verbose=True
    )

    print(f"\nDécodeur ONNX exporté avec succès à: {onnx_file}")
    print(f"Shape d'entrée attendue: {dummy_input.shape}")
    print(f"Shape de sortie: {model(dummy_input).shape}")

weights_file = "decoder.pth"
onnx_file = "decoder.onnx"

export_decoder(weights_file, onnx_file)