# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania

"""Optical simulation for the PhotonEdge 4f Fourier-plane architecture.

Implements DoG kernel generation, phase drift modeling, and optical
propagation through the 4f correlator system.
"""

import functools
import numpy as np


@functools.lru_cache(maxsize=16)
def _fft_kernel_cached(image_size: int, ksize: int,
                       sigma1: float, sigma2: float) -> np.ndarray:
    """Cached FFT of the DoG kernel — avoids recomputing H(u,v) for repeated calls."""
    kernel = dog_kernel_embedded(image_size, ksize, sigma1, sigma2)
    return np.fft.fft2(np.fft.ifftshift(kernel))


def dog_kernel_embedded(image_size: int, ksize: int,
                        sigma1: float, sigma2: float) -> np.ndarray:
    """Generate a Difference-of-Gaussians kernel embedded in an image-sized array.

    The kernel is zero-DC (mean subtracted) and placed at the center of the
    output array for correct FFT-based convolution.

    Args:
        image_size: Size of the output array (square).
        ksize: Size of the DoG kernel (must be odd).
        sigma1: Standard deviation of the narrow Gaussian (passband high-freq).
        sigma2: Standard deviation of the wide Gaussian (passband low-freq).

    Returns:
        2D array of shape (image_size, image_size) with the DoG kernel centered.
    """
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    x, y = np.meshgrid(ax, ax)
    d2 = x * x + y * y

    g1 = np.exp(-d2 / (2 * sigma1 * sigma1))
    g2 = np.exp(-d2 / (2 * sigma2 * sigma2))
    g1 /= (g1.sum() + 1e-12)
    g2 /= (g2.sum() + 1e-12)

    dog = g1 - g2
    dog -= dog.mean()  # enforce zero-DC

    kernel = np.zeros((image_size, image_size), dtype=np.float32)
    c = image_size // 2
    h = ksize // 2
    kernel[c - h:c + h + 1, c - h:c + h + 1] = dog.astype(np.float32)
    return kernel


def optical_sim_linear(img: np.ndarray, kernel: np.ndarray,
                       snr_db: float, drift_std: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Simulate the 4f optical system with phase drift and detector noise.

    Models the full optical pipeline:
    1. Input field E_in = I(x,y) * exp(j*phi) where phi ~ N(0, drift_std^2)
    2. Fourier-plane multiplication: F{E_in} * H(u,v)
    3. Inverse Fourier transform
    4. Detector output: Re{E_out} + Gaussian noise at specified SNR

    Args:
        img: Input image (2D, float32, typically binary 0/1).
        kernel: DoG kernel (same size as img, centered).
        snr_db: Signal-to-noise ratio in dB.
        drift_std: Standard deviation of phase drift (radians).
        rng: NumPy random number generator for reproducibility.

    Returns:
        Noisy detector output Y(x,y) as float32 array.
    """
    phase = rng.normal(0.0, drift_std, img.shape)
    E_in = img.astype(np.float32) * np.exp(1j * phase)

    F_in = np.fft.fft2(E_in)
    H = np.fft.fft2(np.fft.ifftshift(kernel))  # consider _fft_kernel_cached for repeated calls
    E_out = np.fft.ifft2(F_in * H)

    Y = np.real(E_out).astype(np.float32)

    sig_power = float(np.mean(Y * Y))
    sig_power = max(sig_power, 1e-12)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), Y.shape).astype(np.float32)

    return (Y + noise).astype(np.float32)
