"""FingerChain 实验的块状 DCT/IDCT 工具，支持多通道图像并区分频带。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

BLOCK_SIZE = 8  # 论文采用 8x8 DCT，这里固定块尺寸
FREQ_MIN_SUM = 4  # 仅在 u+v >= 4 的系数上嵌入，避免低频

_u = np.arange(BLOCK_SIZE)[:, None]
_x = np.arange(BLOCK_SIZE)[None, :]
_alpha = np.array([1 / math.sqrt(2.0)] + [1.0] * (BLOCK_SIZE - 1))
_COS_TABLE = np.cos((math.pi * (2 * _x + 1) * _u) / (2 * BLOCK_SIZE))
_DCT_MATRIX = 0.5 * (_alpha[:, None] * _COS_TABLE)


def _block_dct(block: np.ndarray) -> np.ndarray:
    """对单个 8x8 块执行 DCT，使用矩阵乘法加速。"""
    block = block.astype(np.float64) - 128.0
    temp = np.einsum("ij,jk->ik", _DCT_MATRIX, block)
    return np.einsum("ij,jk->ik", temp, _DCT_MATRIX.T)


def _block_idct(coeffs: np.ndarray) -> np.ndarray:
    """执行 IDCT，生成的像素值裁剪到 [0,255]。"""
    temp = np.einsum("ij,jk->ik", _DCT_MATRIX.T, coeffs)
    block = np.einsum("ij,jk->ik", temp, _DCT_MATRIX) + 128.0
    return np.clip(block, 0, 255)


@dataclass
class DCTVectorBundle:
    """封装多个通道的 DCT 信息。"""

    padded_shapes: List[Tuple[int, int]]
    original_shapes: List[Tuple[int, int]]
    slices: List[slice]
    mask: np.ndarray

    def reconstruct(self, vector: np.ndarray) -> np.ndarray:
        """根据向量还原出图像，自动拼接通道。"""
        channels = []
        for idx, channel_slice in enumerate(self.slices):
            channel_vec = vector[channel_slice]
            channel_img = dct_vector_to_image(channel_vec, self.padded_shapes[idx], self.original_shapes[idx])
            channels.append(channel_img)
        if len(channels) == 1:
            return channels[0]
        return np.stack(channels, axis=-1)

    def length(self) -> int:
        return self.slices[-1].stop if self.slices else 0

    def clone(self) -> "DCTVectorBundle":
        return DCTVectorBundle(
            padded_shapes=list(self.padded_shapes),
            original_shapes=list(self.original_shapes),
            slices=[slice(s.start, s.stop, s.step) for s in self.slices],
            mask=self.mask.copy(),
        )


def _block_frequency_mask() -> np.ndarray:
    """生成单个 8x8 块的频域掩码，跳过低频位置。"""
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
    for u in range(BLOCK_SIZE):
        for v in range(BLOCK_SIZE):
            if u + v >= FREQ_MIN_SUM:
                mask[u, v] = 1.0
    return mask


_BLOCK_MASK = _block_frequency_mask().reshape(-1)


def image_to_dct_vector(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    if image.ndim != 2:
        raise ValueError("Only single-channel grayscale images are supported")
    orig_shape = image.shape
    # 以边界像素填充到 8 的倍数，避免残块
    pad_h = (BLOCK_SIZE - image.shape[0] % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - image.shape[1] % BLOCK_SIZE) % BLOCK_SIZE
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    padded_shape = padded.shape
    blocks = []
    masks = []
    for i in range(0, padded.shape[0], BLOCK_SIZE):
        for j in range(0, padded.shape[1], BLOCK_SIZE):
            block = padded[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
            blocks.append(_block_dct(block).reshape(-1))
            masks.append(_BLOCK_MASK)
    vector = np.concatenate(blocks).astype(np.float64)
    mask_vec = np.concatenate(masks).astype(np.float64)
    return vector, mask_vec, padded_shape, orig_shape


def dct_vector_to_image(vector: np.ndarray, padded_shape: Tuple[int, int], original_shape: Tuple[int, int]) -> np.ndarray:
    """把 DCT 系数向量还原为图像，最后剪裁到原始尺寸。"""
    expected = (padded_shape[0] // BLOCK_SIZE) * (padded_shape[1] // BLOCK_SIZE)
    block_vectors = np.split(vector, expected)
    reconstructed = np.zeros(padded_shape, dtype=np.float64)
    idx = 0
    for i in range(0, padded_shape[0], BLOCK_SIZE):
        for j in range(0, padded_shape[1], BLOCK_SIZE):
            coeffs = block_vectors[idx].reshape(BLOCK_SIZE, BLOCK_SIZE)
            block = _block_idct(coeffs)
            reconstructed[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE] = block
            idx += 1
    cropped = reconstructed[: original_shape[0], : original_shape[1]]
    return cropped.round().astype(np.uint8)


def image_to_coeff_bundle(image: np.ndarray) -> Tuple[np.ndarray, DCTVectorBundle]:
    """将灰度或彩色图像映射到统一的系数向量。"""
    if image.ndim == 2:
        vector, mask, padded_shape, original_shape = image_to_dct_vector(image)
        bundle = DCTVectorBundle(
            padded_shapes=[padded_shape],
            original_shapes=[original_shape],
            slices=[slice(0, len(vector))],
            mask=mask,
        )
        return vector.copy(), bundle
    if image.ndim == 3:
        vectors = []
        slices = []
        masks = []
        padded_shapes = []
        original_shapes = []
        start = 0
        for channel in range(image.shape[2]):
            vec, mask_vec, padded_shape, original_shape = image_to_dct_vector(image[:, :, channel])
            vectors.append(vec)
            masks.append(mask_vec)
            end = start + len(vec)
            slices.append(slice(start, end))
            padded_shapes.append(padded_shape)
            original_shapes.append(original_shape)
            start = end
        vector = np.concatenate(vectors)
        mask_vec = np.concatenate(masks)
        bundle = DCTVectorBundle(
            padded_shapes=padded_shapes,
            original_shapes=original_shapes,
            slices=slices,
            mask=mask_vec,
        )
        return vector, bundle
    raise ValueError("Unsupported image dimensions for DCT conversion")
