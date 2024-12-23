# Image Denoising Comparison: Total Variation vs. DnCNN

## Purpose

The inspiration for this project came from coursework where we applied Total Variation denoising and learned about the Barzilai-Borwein (BB) optimization method for image denoising. I wanted to see if I could find a new denoising method online that would surpass the Total Variation (TV) approach. This led me to explore the DnCNN model from the paper “Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising” by Kai Zhang et al. In this project, I compare the performance of classical TV denoising with the deep learning–based DnCNN to see how well modern neural networks can handle image denoising compared to traditional techniques.


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

The **DnCNN method** is expected to handle complex noise patterns effectively, often achieving higher PSNR values compared to the **TV method**, which may be better at preserving edges but less capable with complex noise.

These results demonstrate that DnCNN generally produces higher PSNR values, indicating better denoising performance, especially on complex noise patterns.

## Instructions for Execution

### Total Variation Denoising (TV Denoising)

To execute TV denoising on an image file, use the following command:

```bash
python tv_denoising.py input_image.png output_image.png --target_size 256 256 --lambd 1.0 --max_iterations 500 --eps 1e-8
```

To execute DnCNN denoising on an image file, use the following command:

```bash
python dncnn_denoising.py input_image.png output_image.png --weights dncnn_model.pth --target_size 256 256 --contrast_factor 1.15
```

---
