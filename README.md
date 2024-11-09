# Image Denoising Comparison: Total Variation vs. DnCNN

This project compares two approaches to image denoising: the **Total Variation (TV) method** (using the Barzilai-Borwein gradient descent) and the **Deep Convolutional Neural Network (DnCNN)** method, as proposed in ["Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" by Kai Zhang et al.](https://arxiv.org/abs/1608.03981).

## Methods Overview

1. **Total Variation (TV) Method**:
   - Implements a gradient descent optimization using the Barzilai-Borwein method.
   - Optimizes for smoothness by reducing the total variation in the image.
   - Suitable for removing noise while preserving edges.

2. **DnCNN (Deep CNN)**:
   - A neural network-based approach that directly learns the residual noise and subtracts it from the noisy image.
   - Uses a residual learning framework, making it effective for complex noise patterns.
   - Trained using residual mappings for more accurate noise estimation.

## Evaluation Metric

To assess denoising quality, we use **Peak Signal-to-Noise Ratio (PSNR)**. A higher PSNR value indicates a cleaner, more accurate denoised image.

### Example Comparisons
![Example1](comparison_results/comparison_1.png)
![Example2](comparison_results/comparison_2.png)


Repeat this row for each example saved in the `comparison_results` directory.

The **DnCNN method** is expected to handle complex noise patterns effectively, often achieving higher PSNR values compared to the **TV method**, which may be better at preserving edges but less capable with complex noise.

These results demonstrate that DnCNN generally produces higher PSNR values, indicating better denoising performance, especially on complex noise patterns.

---
