import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import argparse

# Define DnCNN model
class DnCNN(nn.Module):
    def __init__(self, depth=17, num_channels=1):
        super(DnCNN, self).__init__()
        
        layers = []
        
        # (i) First layer: Conv + ReLU
        layers.append(nn.Conv2d(num_channels, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # (ii) Intermediate layers: Conv + BatchNorm + ReLU
        for _ in range(1, depth - 1):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        
        # (iii) Last layer: Conv (no activation here)
        layers.append(nn.Conv2d(64, num_channels, kernel_size=3, padding=1, bias=True))
        
        # Stack layers
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, y):
        # Predict the noise
        residual = self.dncnn(y)
        # Denoised image is noisy input - predicted noise
        denoised_image = y - residual
        return denoised_image, residual

# Load model from weights
def load_model(weights_path, device):
    model = DnCNN(num_channels=1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load and resize image
def load_and_resize_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert("L")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # Add batch dimension

# Denoise image with DnCNN
def denoise_image_dncnn(model, noisy_image, contrast_factor=1.15):
    device = next(model.parameters()).device
    noisy_image = noisy_image.to(device)
    denoised_image, _ = model(noisy_image)
    denoised_image = torch.clamp(denoised_image, 0, 1)
    denoised_image = F.adjust_contrast(denoised_image.squeeze(0).cpu(), contrast_factor)
    return denoised_image

# Save image to specified path
def save_image(tensor, output_path):
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image.save(output_path)

# Main function for denoising a single image
def denoise_single_image(input_path, output_path, model, target_size=(256, 256), contrast_factor=1.15):
    # Load and denoise image
    noisy_image = load_and_resize_image(input_path, target_size=target_size)
    denoised_image = denoise_image_dncnn(model, noisy_image, contrast_factor=contrast_factor)

    # Save denoised image
    save_image(denoised_image, output_path)
    print(f"Denoised image saved to: {output_path}")

# Argument parser for command-line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a single image using the DnCNN model")
    parser.add_argument("input_image", type=str, help="Path to the noisy image file")
    parser.add_argument("output_image", type=str, help="Path to save the denoised image")
    parser.add_argument("--weights", type=str, default="dncnn_model.pth", help="Path to DnCNN model weights")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256], help="Target image size")
    parser.add_argument("--contrast_factor", type=float, default=1.15, help="Contrast adjustment factor")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    
    denoise_single_image(args.input_image, args.output_image, model, target_size=tuple(args.target_size), contrast_factor=args.contrast_factor)
