import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

def load_and_resize_image(image_path, target_size=(256, 256)):
    """
    Load an image from a specified path, convert it to grayscale, 
    and resize it to the target size if necessary.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): The desired image size in pixels (width, height).
        
    Returns:
        torch.Tensor: Resized image as a tensor.
    """
    # Load the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale

    # Resize the image to target size if it's not already the correct size
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)  # Use LANCZOS filter for resizing
    
    # Convert to tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    
    return image_tensor

def D_mat(N):
    """Creates a differentiation matrix as a PyTorch tensor."""
    I = torch.eye(N)
    D = I - torch.roll(I, shifts=-1, dims=1)
    D[-1, -1] = 0  # Ensure the last row is zero to handle boundary
    return D

def grad(v, image, lambd=1.0):
    """
    Computes the gradient for the cost function using Total Variation.
    
    Args:
        v (torch.Tensor): Current image tensor (1, H, W).
        image (torch.Tensor): Original noisy image tensor.
        lambd (float): Regularization parameter.
    
    Returns:
        torch.Tensor: Gradient of the cost function with shape (1, H, W).
    """
    v = v.squeeze(0)  # Shape now [H, W]
    image = image.squeeze(0)  # Shape now [H, W]
    
    N_L, N_R = v.shape
    L = D_mat(N_L).to(v.device)
    R = D_mat(N_R).to(v.device)
    
    grad_value = (v - image) + lambd * ((L.T @ L @ v) + v @ R.T @ R)
    
    return grad_value.unsqueeze(0)  # Add back channel dimension

def cost(v, image, lambd=1.0):
    """
    Cost function for Total Variation (TV) denoising.

    Args:
        v (torch.Tensor): Current image tensor (denoised estimate).
        image (torch.Tensor): Noisy input image tensor.
        lambd (float): Regularization parameter for TV.

    Returns:
        torch.Tensor: Cost value (scalar).
    """
    v = v.squeeze(0)
    image = image.squeeze(0)
    
    N_L, N_R = v.shape
    L = D_mat(N_L).to(v.device)
    R = D_mat(N_R).to(v.device)
    
    diff_v = (L @ (v - image)) ** 2 + (v - image) @ R.T @ R
    tv_term = torch.sum(diff_v)
    
    return lambd * 0.5 * tv_term

def BBstep(grad_f, x, xm1, signal):
    """
    Computes the Barzilai-Borwein step size.

    Args:
        grad_f: Function to compute the gradient of f at a given x.
        x: Current point.
        xm1: Previous point.
        signal: The initial signal for calculating gradients.

    Returns:
        Δ: Step size computed using Barzilai-Borwein method.
    """
    grad_diff = grad_f(x, signal) - grad_f(xm1, signal)
    x_diff = x - xm1

    # Flatten the tensors to 1D for dot product calculation
    grad_diff_flat = grad_diff.view(-1)
    x_diff_flat = x_diff.view(-1)

    Δ = (grad_diff_flat @ x_diff_flat) / (grad_diff_flat @ grad_diff_flat)
    
    return Δ


def BB_fixed_step(grad, signal, f, max_iterations=500, eps=1e-8):
    """
    Performs the Barzilai-Borwein fixed-step optimization.
    
    Args:
        grad: Function to compute the gradient of f at a given x.
        signal: The initial signal to start with.
        f: Function to compute the cost at a given x.
        max_iterations: Maximum number of iterations for the optimization.
        eps: Convergence threshold for the optimization.
    
    Returns:
        Final optimized tensor, number of iterations, and all intermediate iterations.
    """
    x0 = signal.clone()
    iterations = [x0]
    G = [grad(x0, signal)]
    f_values = [f(x0, signal)]

    I = torch.eye(x0.shape[1]).to(x0.device)

    iterations.append(iterations[0] - 0.05 * G[0])
    G.append(grad(iterations[1], signal))
    f_values.append(f(iterations[1], signal))
    i = 1

    while (torch.abs(f_values[i] - f_values[i-1]) > eps) and (i < max_iterations):
        i += 1
        Δ = BBstep(grad, iterations[i-1], iterations[i-2], signal)
        iterations.append(iterations[i-1] - Δ * I @ G[i-1])
        f_values.append(f(iterations[i], signal))
        G.append(grad(iterations[i], signal))

    iterations = torch.stack(iterations)
    return iterations[-1], i, iterations



def main_tv_denoising(image_path, output_path, target_size=(256, 256), lambd=1.0, max_iterations=500, eps=1e-8):
    """
    Main function for Total Variation Denoising using the Barzilai-Borwein method.
    
    Args:
        image_path (str): Path to the noisy image.
        output_path (str): Path to save the denoised image.
        target_size (tuple): Size to which the image should be resized.
        lambd (float): Regularization parameter for TV.
        max_iterations (int): Max number of iterations.
        eps (float): Convergence threshold for BB method.
    """
    # Load and resize image
    noisy_image = load_and_resize_image(image_path, target_size)
    
    # Perform denoising with Barzilai-Borwein method
    denoised_image, _, _ = BB_fixed_step(grad, noisy_image, cost, max_iterations=max_iterations, eps=eps)
    
    # Clamp to [0, 1] and convert to PIL format
    denoised_image = torch.clamp(denoised_image, 0, 1)
    denoised_image_pil = transforms.ToPILImage()(denoised_image.squeeze(0))
    
    # Save or show the output
    denoised_image_pil.save(output_path)
    print(f"Denoised image saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Total Variation Denoising using Barzilai-Borwein Method")
    parser.add_argument("input_path", type=str, help="Path to the noisy input image")
    parser.add_argument("output_path", type=str, help="Path to save the denoised image")
    parser.add_argument("--target_size", type=int, nargs=2, default=(256, 256), help="Target size for resizing the image")
    parser.add_argument("--noise_level", type=int, default=25, help="Noise level for denoising (over 255)")
    parser.add_argument("--lambd", type=float, default=1.0, help="Regularization parameter for TV")
    parser.add_argument("--max_iterations", type=int, default=500, help="Max number of iterations")
    parser.add_argument("--eps", type=float, default=1e-8, help="Convergence threshold for BB method")

    args = parser.parse_args()

    main_tv_denoising(
        args.input_path,
        args.output_path,
        target_size=tuple(args.target_size),
        lambd=args.lambd,
        max_iterations=args.max_iterations,
        eps=args.eps
    )