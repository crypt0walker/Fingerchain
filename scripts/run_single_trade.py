"""执行一次 1 对 1 的 FingerChain 交易并统计各环节耗时。"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np

# 将仓库根目录加入 sys.path，方便直接运行脚本
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fingerchain.chain.mock_eth import MockEthChainAPI
from fingerchain.crypto.paillier import generate_keypair
from fingerchain.fp.lut_scheme import (
    MediaEncryptionResult,
    extract_fingerprint_bits,
    generate_e_lut,
    generate_g_matrix,
    owner_compute_d_lut,
    owner_encrypt_media,
    user_decrypt_d_lut,
    user_prepare_registration,
    user_recover_image,
)

DEFAULT_IMAGE_SIZE = 256
DEFAULT_FINGERPRINT_LEN = 128
LUT_LEN = 1_000
SIGMA_E = 1e3
SIGMA_W = 0.1
FANOUT_S = 2
KEY_BITS = 1024
DEFAULT_IMAGE = REPO_ROOT / "Lenna.jpg"


def _hash_ciphertexts(ciphertexts) -> str:
    """对一组密文求哈希，模拟链上的承诺值。"""
    hasher = hashlib.sha256()
    for value in ciphertexts:
        byte_len = max(1, (value.bit_length() + 7) // 8)
        hasher.update(value.to_bytes(byte_len, "big", signed=False))
    return hasher.hexdigest()


def _ciphertexts_size(ciphertexts) -> int:
    """估算加密 payload 的字节数，用于通信开销统计。"""
    total = 0
    for value in ciphertexts:
        total += max(1, (value.bit_length() + 7) // 8)
    return total


def _psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """计算 PSNR 评估水印图像质量。"""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


@dataclass
class SingleTradeMetrics:
    """封装一次交易涉及的关键指标。"""

    owner_encrypt_media: float
    owner_compute_d_lut: float
    user_decrypt_d_lut: float
    user_recover_image: float
    mock_share_tx: float
    psnr: float
    payload_bytes_owner_to_user: int
    encrypted_media_bytes: int
    watermarked_output: str
    fingerprint_bit_errors: int


def _temp_path(suffix: str) -> Path:
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(tmp)
    path.unlink(missing_ok=True)
    return path


def _run_sips_convert(src: Path, dst: Path, target_size: int | None, fmt: str | None) -> None:
    """调用 sips 进行格式转换及可选缩放。"""
    cmd = ["sips"]
    if fmt:
        cmd += ["-s", "format", fmt]
    if target_size:
        cmd += ["--resampleHeightWidth", str(target_size), str(target_size)]
    cmd += [str(src), "--out", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_input_image(image_path: Path, size: int | None) -> np.ndarray:
    """通过 sips 转换到 BMP，然后解析为 RGB numpy 数组。"""
    tmp_bmp = _temp_path(".bmp")
    try:
        _run_sips_convert(image_path, tmp_bmp, size, "bmp")
        return _read_bmp_rgb(tmp_bmp)
    finally:
        tmp_bmp.unlink(missing_ok=True)


def _read_bmp_rgb(bmp_path: Path) -> np.ndarray:
    """解析 24/32bit BMP 并转换为 RGB 矩阵。"""
    data = bmp_path.read_bytes()
    if data[:2] != b"BM":
        raise ValueError("Unsupported BMP header")
    pixel_offset = int.from_bytes(data[10:14], "little")
    dib_size = int.from_bytes(data[14:18], "little")
    if dib_size < 40:
        raise ValueError("Unsupported BMP DIB header")
    width = int.from_bytes(data[18:22], "little", signed=True)
    height = int.from_bytes(data[22:26], "little", signed=True)
    planes = int.from_bytes(data[26:28], "little")
    bpp = int.from_bytes(data[28:30], "little")
    compression = int.from_bytes(data[30:34], "little")
    if planes != 1 or compression != 0:
        raise ValueError("Only uncompressed BMP is supported")
    if bpp not in (24, 32):
        raise ValueError(f"Unsupported BMP bits per pixel: {bpp}")
    abs_width = abs(width)
    abs_height = abs(height)
    channels = bpp // 8
    row_stride = ((bpp * abs_width + 31) // 32) * 4
    pixel_data = memoryview(data)[pixel_offset:]
    rows = []
    for row_idx in range(abs_height):
        start = row_idx * row_stride
        row = np.frombuffer(pixel_data[start : start + abs_width * channels], dtype=np.uint8)
        row = row.reshape((abs_width), channels)
        rows.append(row[:, :3])  # BGR
    array = np.stack(rows, axis=0)
    if height > 0:  # bottom-up, 需要翻转
        array = array[::-1]
    rgb = array[:, :, ::-1]  # BGR -> RGB
    return rgb.astype(np.uint8)


def _write_raw_image(array: np.ndarray, path: Path) -> None:
    """根据通道数输出 P5(PGM) 或 P6(PPM)。"""
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        header = f"P5\n{array.shape[1]} {array.shape[0]}\n255\n".encode()
        data = array.tobytes()
    elif array.ndim == 3 and array.shape[2] == 3:
        header = f"P6\n{array.shape[1]} {array.shape[0]}\n255\n".encode()
        data = array.tobytes()
    else:
        raise ValueError("Unsupported image shape for saving")
    with open(path, "wb") as f:
        f.write(header)
        f.write(data)


def save_image(array: np.ndarray, output_path: Path) -> None:
    """保存嵌入指纹后的图像，如需 PNG/JPG 则借助 sips 转换。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".pgm", ".ppm"}:
        _write_raw_image(array, output_path)
        return
    tmp_raw = _temp_path(".ppm")
    try:
        _write_raw_image(array, tmp_raw)
        fmt_map = {
            ".png": "png",
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".tif": "tiff",
            ".tiff": "tiff",
            ".bmp": "bmp",
        }
        target_fmt = fmt_map.get(suffix)
        if not target_fmt:
            fallback = output_path.with_suffix(".ppm")
            _write_raw_image(array, fallback)
            return
        _run_sips_convert(tmp_raw, output_path, None, target_fmt)
    finally:
        tmp_raw.unlink(missing_ok=True)


def prompt_positive_int(message: str, default: int, allow_zero: bool = False) -> int:
    """交互式输入正整数，回车使用默认值，可选允许 0。"""
    while True:
        user_input = input(f"{message} (默认 {default}{'，输入0表示保持原尺寸' if allow_zero else ''}): ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
        except ValueError:
            print("请输入合法的整数。")
            continue
        if allow_zero and value == 0:
            return 0
        if value <= 0:
            print("请输入大于 0 的整数。")
            continue
        return value


def run_single_trade(image: np.ndarray, fingerprint_length: int, rng: np.random.Generator, output_path: Path) -> Dict[str, float]:
    """串联一次完整交易流程并返回指标。"""

    chain = MockEthChainAPI()
    owner_addr = "0xowner"
    user_addr = "0xuser"

    user_pub, user_priv = generate_keypair(bits=KEY_BITS)  # 用户 Paillier 密钥
    judge_pub, _ = generate_keypair(bits=KEY_BITS)  # 法官密钥仅用于演示

    t_u1_start = time.perf_counter()
    user_material = user_prepare_registration(fingerprint_length, user_pub, judge_pub, rng)
    t_u1_end = time.perf_counter()
    print(f"[user] U1_encrypt_fingerprint: {t_u1_end - t_u1_start:.6f}s")
    fp_commit_user = _hash_ciphertexts(user_material.enc_for_owner)
    fp_commit_judge = _hash_ciphertexts(user_material.enc_for_judge)

    # Mock 注册 owner/user，真实环境可替换为 web3 调用
    chain.register_owner({"address": owner_addr, "pkOwner": "owner-pk"})
    chain.register_user(
        {
            "address": user_addr,
            "pkUser": "user-pk",
            "fpCommitUser": fp_commit_user,
            "fpCommitJudge": fp_commit_judge,
        }
    )

    t_elut0 = time.perf_counter()
    e_lut = generate_e_lut(LUT_LEN, SIGMA_E, rng)
    t_elut1 = time.perf_counter()
    print(f"[owner] E-LUT generation time: {t_elut1 - t_elut0:.6f}s")
    g_matrix = generate_g_matrix(LUT_LEN, fingerprint_length, rng)
    session_seed = rng.bytes(16)  # 控制 B_m 采样

    t0 = time.perf_counter()
    media_result: MediaEncryptionResult = owner_encrypt_media(image, e_lut, session_seed, FANOUT_S)
    t_pack0 = time.perf_counter()
    media_package = media_result.to_share_package()
    owner_media_payload = media_package.encrypted_coeffs.nbytes + media_package.positions.nbytes
    t_pack1 = time.perf_counter()
    print(
        f"[owner] O6_pack_and_send (media): {t_pack1 - t_pack0:.6f}s | payload={owner_media_payload} bytes"
    )
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    enc_d_lut = owner_compute_d_lut(e_lut, g_matrix, user_material.enc_for_owner, user_pub, SIGMA_W)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    d_lut = user_decrypt_d_lut(enc_d_lut, user_priv)
    t5 = time.perf_counter()
    print(f"[user] U2_decrypt_D_LUT: {t5 - t4:.6f}s")

    t6 = time.perf_counter()
    recovered_image, recovered_coeffs = user_recover_image(
        media_package.encrypted_coeffs,
        d_lut,
        media_package.positions,
        media_package.bundle,
    )
    save_image(recovered_image, output_path)
    t7 = time.perf_counter()

    t8 = time.perf_counter()
    chain.share_media({"media_key": "media-1", "owner": owner_addr, "user": user_addr})
    t9 = time.perf_counter()

    delta_coeffs = np.nan_to_num(recovered_coeffs - media_result.base_coeffs, nan=0.0, posinf=0.0, neginf=0.0)
    t_fp0 = time.perf_counter()
    extracted_bits = extract_fingerprint_bits(delta_coeffs, media_result.positions, g_matrix)
    t_fp1 = time.perf_counter()
    print(f"[user] U5_fingerprint_extraction: {t_fp1 - t_fp0:.6f}s")
    bit_errors = int(np.sum(extracted_bits != user_material.bits))

    dlut_payload_bytes = _ciphertexts_size(enc_d_lut)
    total_payload_bytes = owner_media_payload + dlut_payload_bytes
    print(
        f"[owner] O6_pack_and_send (total payload incl. D-LUT): {total_payload_bytes} bytes"
    )

    metrics = SingleTradeMetrics(
        owner_encrypt_media=t1 - t0,
        owner_compute_d_lut=t3 - t2,
        user_decrypt_d_lut=t5 - t4,
        user_recover_image=t7 - t6,
        mock_share_tx=t9 - t8,
        psnr=_psnr(image, recovered_image),
        payload_bytes_owner_to_user=dlut_payload_bytes,
        encrypted_media_bytes=media_result.encrypted_coeffs.astype(np.float64).nbytes,
        watermarked_output=str(output_path),
        fingerprint_bit_errors=bit_errors,
    )
    return asdict(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single FingerChain trade prototype")
    parser.add_argument("--seed", type=int, default=2024, help="Seed for reproducibility")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="路径，默认使用仓库中的 Lenna.jpg")
    parser.add_argument("--image-size", type=int, default=None, help="将图像缩放到指定尺寸（正方形），输入 0 表示保持原尺寸；默认交互式输入，预设 256")
    parser.add_argument("--fingerprint-length", type=int, default=None, help="嵌入指纹的比特长度；默认交互式输入，预设 128")
    parser.add_argument("--output", type=str, default="outputs/watermarked.png", help="保存嵌入指纹后的图片路径")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到指定的图像文件: {image_path}")
    size = args.image_size
    if size is not None and size < 0:
        raise ValueError("image-size must be >= 0")
    if size is None:
        size = prompt_positive_int("请输入图像像素边长", DEFAULT_IMAGE_SIZE, allow_zero=True)
    if size == 0:
        resize_to = None
    else:
        resize_to = size
    fprint_len = args.fingerprint_length
    if fprint_len is not None and fprint_len <= 0:
        raise ValueError("fingerprint-length must be > 0")
    if fprint_len is None:
        fprint_len = prompt_positive_int("请输入指纹比特长度", DEFAULT_FINGERPRINT_LEN)
    image = load_input_image(image_path, resize_to)
    metrics = run_single_trade(image, fprint_len, rng, Path(args.output))
    print("Single trade metrics:")
    time_keys = {
        "owner_encrypt_media",
        "owner_compute_d_lut",
        "user_decrypt_d_lut",
        "user_recover_image",
        "mock_share_tx",
    }
    for key, value in metrics.items():
        if key in time_keys:
            print(f"  {key}: {value:.6f} s")
        elif key.endswith("bytes"):
            print(f"  {key}: {value} bytes")
        elif key == "watermarked_output":
            print(f"  {key}: {value}")
        elif key == "fingerprint_bit_errors":
            print(f"  {key}: {int(value)} bit errors")
        else:
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
