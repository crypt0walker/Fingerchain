"""FingerChain 方案中的 LUT 式非对称指纹辅助函数。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import time
from typing import List, Sequence

import numpy as np

from fingerchain.crypto.paillier import (
    PaillierPrivateKey,
    PaillierPublicKey,
    decrypt_vector,
    encrypt_vector,
)
from fingerchain.media.dct import DCTVectorBundle, image_to_coeff_bundle

SCALE = 1 << 7  # 量化因子，略小的 SCALE 能降低嵌入能量


def generate_e_lut(length: int, sigma_e: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """生成服从 N(0, σ_E) 的加密 LUT。"""
    rng = rng or np.random.default_rng()
    return rng.normal(loc=0.0, scale=sigma_e, size=length)


def generate_g_matrix(rows: int, cols: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """生成取值为 ±1 的编码矩阵，保证匹配滤波时有正负号差异。"""
    rng = rng or np.random.default_rng()
    matrix = rng.integers(0, 2, size=(rows, cols), dtype=np.int8)
    return matrix * 2 - 1


def user_generate_fingerprint(length: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """用户本地生成二进制指纹序列。"""
    rng = rng or np.random.default_rng()
    return rng.integers(0, 2, size=length, dtype=np.int64)


@dataclass
class UserFingerprintMaterial:
    bits: np.ndarray
    enc_for_owner: List[int]
    enc_for_judge: List[int]


def user_prepare_registration(
    length: int,
    user_pub: PaillierPublicKey,
    judge_pub: PaillierPublicKey,
    rng: np.random.Generator | None = None,
) -> UserFingerprintMaterial:
    """模拟论文中的用户注册，产生指纹及对应密文。"""
    bits = user_generate_fingerprint(length, rng)
    enc_for_owner = encrypt_vector(user_pub, bits.tolist())
    enc_for_judge = encrypt_vector(judge_pub, bits.tolist())
    return UserFingerprintMaterial(bits=bits, enc_for_owner=enc_for_owner, enc_for_judge=enc_for_judge)


def _quantize(values: np.ndarray) -> np.ndarray:
    """把浮点数转为定点整数，便于 Paillier 运算。"""
    return np.round(values * SCALE).astype(int)


def _hash_seed(seed: bytes | bytearray | str) -> int:
    if isinstance(seed, str):
        seed = seed.encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    return int.from_bytes(digest, "big")


def sample_positions(seed: bytes | bytearray | str, length: int, lut_size: int, fanout: int) -> np.ndarray:
    """由会话种子伪随机采样稀疏矩阵 B_m 中每行的非零位置。"""
    rng = np.random.default_rng(_hash_seed(seed))
    positions = np.empty((length, fanout), dtype=np.int32)
    for idx in range(length):
        positions[idx] = rng.choice(lut_size, size=fanout, replace=False)
    return positions


def apply_positions(base: np.ndarray, lut: np.ndarray, positions: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """根据位置把 LUT 累加到基准向量，模拟 c = m + B_m * LUT。"""
    additions = lut[positions].sum(axis=1)
    if mask is not None:
        additions = additions * mask
    return base + additions


@dataclass
class MediaEncryptionResult:
    encrypted_coeffs: np.ndarray
    positions: np.ndarray
    bundle: DCTVectorBundle
    base_coeffs: np.ndarray

    def to_share_package(self) -> "MediaSharePackage":
        return MediaSharePackage(
            encrypted_coeffs=self.encrypted_coeffs.copy(),
            positions=self.positions.copy(),
            bundle=self.bundle.clone(),
        )


@dataclass
class MediaSharePackage:
    encrypted_coeffs: np.ndarray
    positions: np.ndarray
    bundle: DCTVectorBundle


def owner_encrypt_media(
    image: np.ndarray,
    e_lut: np.ndarray,
    seed: bytes,
    fanout: int,
) -> MediaEncryptionResult:
    """Owner 侧：把图像转换到 DCT 向量并执行 Eq.(3) 的加密。"""
    t_total0 = time.perf_counter()
    t_dct0 = time.perf_counter()
    coeff_vector, bundle = image_to_coeff_bundle(image)
    t_dct1 = time.perf_counter()
    t_sample0 = time.perf_counter()
    positions = sample_positions(seed, len(coeff_vector), len(e_lut), fanout)
    t_sample1 = time.perf_counter()
    t_lut0 = time.perf_counter()
    encrypted_coeffs = apply_positions(coeff_vector, e_lut, positions, mask=bundle.mask)
    t_lut1 = time.perf_counter()
    t_total1 = time.perf_counter()
    dct_time = t_dct1 - t_dct0
    sample_time = t_sample1 - t_sample0
    lut_time = t_lut1 - t_lut0
    total_time = t_total1 - t_total0
    tracked = dct_time + sample_time + lut_time
    print(
        f"[owner] DCT: {dct_time:.6f}s | B_m sampling: {sample_time:.6f}s | LUT add: {lut_time:.6f}s | total: {total_time:.6f}s"
    )
    if total_time > 0:
        diff = total_time - tracked
        print(f"[owner] accounted={tracked:.6f}s, unaccounted={diff:.6f}s")
    return MediaEncryptionResult(
        encrypted_coeffs=encrypted_coeffs,
        positions=positions,
        bundle=bundle,
        base_coeffs=coeff_vector,
    )


def user_recover_image(
    encrypted_coeffs: np.ndarray,
    d_lut: np.ndarray,
    positions: np.ndarray,
    bundle: DCTVectorBundle,
) -> tuple[np.ndarray, np.ndarray]:
    """User 侧：使用 D-LUT 恢复图像，相当于 Eq.(4)。"""
    t_total0 = time.perf_counter()
    t_apply0 = time.perf_counter()
    recovered_coeffs = apply_positions(encrypted_coeffs, d_lut, positions, mask=bundle.mask)
    t_apply1 = time.perf_counter()
    t_idct0 = time.perf_counter()
    image = bundle.reconstruct(recovered_coeffs)
    t_idct1 = time.perf_counter()
    t_total1 = time.perf_counter()
    apply_time = t_apply1 - t_apply0
    idct_time = t_idct1 - t_idct0
    total_time = t_total1 - t_total0
    tracked = apply_time + idct_time
    print(f"[user] LUT add: {apply_time:.6f}s | IDCT: {idct_time:.6f}s | total: {total_time:.6f}s")
    if total_time > 0:
        diff = total_time - tracked
        print(f"[user] accounted={tracked:.6f}s, unaccounted={diff:.6f}s")
    return image, recovered_coeffs


def extract_fingerprint_bits(
    delta_coeffs: np.ndarray,
    positions: np.ndarray,
    g_matrix: np.ndarray,
) -> np.ndarray:
    """通过估算 W-LUT 再做相关检测，验证指纹提取正确性。"""
    t_len = g_matrix.shape[0]
    accum = np.zeros(t_len, dtype=np.float64)
    counts = np.zeros(t_len, dtype=np.float64)
    for coeff, row_pos in zip(delta_coeffs, positions):
        accum[row_pos] += coeff
        counts[row_pos] += 1.0
    nonzero = counts > 0
    accum[nonzero] /= counts[nonzero]
    accum = np.nan_to_num(accum, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(accum)):
        raise ValueError("Non-finite values encountered while estimating W-LUT.")
    scale = np.max(np.abs(accum))
    if scale > 0:
        accum /= scale
    g_float = g_matrix.astype(np.float64, copy=False)
    projection = (accum[:, None] * g_float).sum(axis=0)
    return (projection >= 0).astype(int)


def owner_compute_d_lut(
    e_lut: np.ndarray,
    g_matrix: np.ndarray,
    enc_fingerprint: Sequence[int],
    user_pub: PaillierPublicKey,
    sigma_w: float,
) -> List[int]:
    """Owner 侧：在密文域构造 D-LUT，公式对应 Eq.(1)(2)。"""
    t_total0 = time.perf_counter()
    if g_matrix.shape[1] != len(enc_fingerprint):
        raise ValueError("G matrix column count must match fingerprint length")
    e_quantized = _quantize(e_lut)
    sigma_q = int(round(sigma_w * SCALE))

    enc_w = []
    enc_minus_one = user_pub.encrypt_int(-1)
    t_encw0 = time.perf_counter()
    for enc_bit in enc_fingerprint:
        # 计算 σ_W*(2b-1)，使用同态运算避免解密
        two_b = user_pub.homomorphic_mul_const(enc_bit, 2)
        two_b_minus_one = user_pub.homomorphic_add(two_b, enc_minus_one)
        enc_w.append(user_pub.homomorphic_mul_const(two_b_minus_one, sigma_q))
    t_encw1 = time.perf_counter()

    enc_d = []
    t_rows0 = time.perf_counter()
    for row_idx in range(g_matrix.shape[0]):
        acc = user_pub.encrypt_int(-int(e_quantized[row_idx]))
        for col_idx in range(g_matrix.shape[1]):
            sign = int(g_matrix[row_idx, col_idx])
            if sign == 0:
                continue
            contribution = user_pub.homomorphic_mul_const(enc_w[col_idx], sign)
            acc = user_pub.homomorphic_add(acc, contribution)
        enc_d.append(acc)
    t_rows1 = time.perf_counter()
    t_total1 = time.perf_counter()
    encw_time = t_encw1 - t_encw0
    rows_time = t_rows1 - t_rows0
    total_time = t_total1 - t_total0
    tracked = encw_time + rows_time
    print(
        f"[owner] O5_compute_D_LUT: Enc(w) {encw_time:.6f}s | Accumulate rows {rows_time:.6f}s | total {total_time:.6f}s"
    )
    if total_time > 0:
        diff = total_time - tracked
        print(f"[owner] O5 accounted={tracked:.6f}s, unaccounted={diff:.6f}s")
    return enc_d


def user_decrypt_d_lut(enc_d_lut: Sequence[int], priv: PaillierPrivateKey) -> np.ndarray:
    """用户解密 D-LUT，并除以 SCALE 还原浮点表示。"""
    plaintext = np.array(decrypt_vector(priv, enc_d_lut), dtype=np.float64)
    return plaintext / SCALE
