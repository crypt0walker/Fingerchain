"""Sanity tests for FingerChain primitives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from fingerchain.crypto.paillier import generate_keypair
from fingerchain.fp.lut_scheme import (
    generate_e_lut,
    generate_g_matrix,
    owner_compute_d_lut,
    user_decrypt_d_lut,
    owner_encrypt_media,
    user_recover_image,
    user_prepare_registration,
    extract_fingerprint_bits,
)


@dataclass
class SanityResult:
    psnr_sigma_zero: float
    max_d_diff: float
    fingerprint_bit_errors: int


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


def run_sigma_zero_test(image: np.ndarray, lut_len: int, fanout: int) -> float:
    e_lut = generate_e_lut(lut_len, sigma_e=1000.0)
    session_seed = b"seed"
    media = owner_encrypt_media(image, e_lut, session_seed, fanout)
    package = media.to_share_package()
    user_pub, user_priv = generate_keypair(bits=512)
    judge_pub, _ = generate_keypair(bits=512)
    material = user_prepare_registration(length=8, user_pub=user_pub, judge_pub=judge_pub)
    g_matrix = generate_g_matrix(lut_len, len(material.bits))
    enc_d = owner_compute_d_lut(e_lut, g_matrix, material.enc_for_owner, user_pub, sigma_w=0.0)
    d_lut = user_decrypt_d_lut(enc_d, user_priv)
    recovered_image, _ = user_recover_image(package.encrypted_coeffs, d_lut, package.positions, package.bundle)
    return psnr(image, recovered_image)


def run_d_lut_toy_test() -> float:
    rng = np.random.default_rng(2024)
    L = 3
    T = 5
    sigma_e = 10.0
    sigma_w = 0.4
    e = rng.normal(0, sigma_e, size=T)
    g = rng.choice([-1, 1], size=(T, L))
    bits = rng.integers(0, 2, size=L)
    user_pub, user_priv = generate_keypair(bits=512)
    enc_bits = [user_pub.encrypt_int(int(b)) for b in bits]
    enc_d = owner_compute_d_lut(e, g, enc_bits, user_pub, sigma_w)
    d_lut = np.array(user_decrypt_d_lut(enc_d, user_priv))
    w = sigma_w * (2 * bits - 1)
    d_plain = -e + g @ w
    return np.max(np.abs(d_plain - d_lut))


def run_fingerprint_loop(image: np.ndarray, lut_len: int, fanout: int) -> int:
    rng = np.random.default_rng(42)
    user_pub, user_priv = generate_keypair(bits=512)
    judge_pub, _ = generate_keypair(bits=512)
    material = user_prepare_registration(length=64, user_pub=user_pub, judge_pub=judge_pub, rng=rng)
    e_lut = generate_e_lut(lut_len, sigma_e=1000.0)
    g_matrix = generate_g_matrix(lut_len, len(material.bits), rng=rng)
    media = owner_encrypt_media(image, e_lut, b"seed", fanout)
    package = media.to_share_package()
    enc_d = owner_compute_d_lut(e_lut, g_matrix, material.enc_for_owner, user_pub, sigma_w=0.1)
    d_lut = user_decrypt_d_lut(enc_d, user_priv)
    recovered_image, recovered_coeffs = user_recover_image(
        package.encrypted_coeffs, d_lut, package.positions, package.bundle
    )
    delta = recovered_coeffs - media.base_coeffs
    extracted = extract_fingerprint_bits(delta, media.positions, g_matrix)
    errors = int(np.sum(extracted != material.bits))
    assert psnr(image, recovered_image) > 30.0
    return errors


def main() -> None:
    image = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
    psnr_zero = run_sigma_zero_test(image, lut_len=256, fanout=2)
    d_diff = run_d_lut_toy_test()
    errors = run_fingerprint_loop(image, lut_len=256, fanout=2)
    res = SanityResult(psnr_sigma_zero=psnr_zero, max_d_diff=d_diff, fingerprint_bit_errors=errors)
    print(res)


if __name__ == "__main__":
    main()
