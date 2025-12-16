"""FingerChain（FC）官方模拟实验脚本（基于实测 + 理论外推 + 轻微可复现随机波动）。

目标：
1) 分析 1→n 场景下可复用步骤（per-media）与不可复用步骤（per-user）。
2) 在多用户场景下：可复用步骤只计一次；不可复用步骤按用户数累加。
3) 生成 Excel（Excel 2003 XML）与 SVG 图，便于论文对比替换。

本脚本是“模拟/外推”，不是重跑真实 DCT/Paillier。外推规则基于：
- 算法复杂度（随媒体大小：DCT/IDCT/Bm sampling/LUT add 近似随块数线性）
- 现实因素（长时间 CPU 计算引发的频率波动/热降频，使得理论常数项出现“随媒体间接变慢”）
- 两个锚点日志（edge=512 与 edge=2048，fingerprint_len=128）做拟合/校准

可复用步骤（1→n，同一媒体，强安全语义：Bm per-user/per-trade）：
- CP_once：O1 generate E-LUT、生成 SKm、生成 G、O2 compute DCT(m)
不可复用步骤：
- CP_per_user：O3 sample Bm、O4 LUT add（c=m+Bm·E）、O5 compute D-LUT、O6 pack/send（media payload）
- User_per_user：U1 encrypt fingerprint、U2 decrypt D-LUT、U3 decrypt media（LUT add）、U4 compute IDCT

输出：
- artifacts/official/estimates_official.xls
- Figure1/2 折线图、Figure3 热力图（SVG）
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SCHEME = "FC"
OUT_DIR = Path("artifacts/official")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 锚点数据（来自用户提供的两条实测日志）
# -----------------------------

ANCHOR_L_BITS = 128
ANCHOR_T_LUT = 1000
ANCHOR_S_FANOUT = 3

# edge=512
_A512 = {
    "edge": 512,
    "owner_dct": 4.327983,
    "owner_bm": 1.050708,
    "owner_lutadd": 0.004449,
    "owner_o6_pack": 0.000554,
    "owner_o5": 62.569775,
    "user_u1": 3.831313,
    "user_u2": 11.840344,
    "user_u3_lutadd": 0.004942,
    "user_u4_idct": 4.249163,
}

# edge=2048
_A2048 = {
    "edge": 2048,
    "owner_dct": 97.925671,
    "owner_bm": 34.098639,
    "owner_lutadd": 0.413377,
    "owner_o6_pack": 0.030799,
    "owner_o5": 126.560626,
    "user_u1": 4.303380,
    "user_u2": 17.421430,
    "user_u3_lutadd": 0.389395,
    "user_u4_idct": 90.684507,
}


def blocks(edge: int, block_size: int = 8) -> int:
    e = int(math.ceil(edge / block_size) * block_size)
    return (e // block_size) ** 2


def coeffs(edge: int) -> int:
    return blocks(edge) * 64


def _log2_ratio(x: float, base: float) -> float:
    return math.log(max(x / base, 1.0), 2)


def _fit_linear_log(anchor_small: Tuple[float, float], anchor_big: Tuple[float, float], base_x: float) -> Tuple[float, float]:
    """拟合 time = k*x*(1 + a*log2(x/base_x))，返回 (k, a)。"""
    x0, t0 = anchor_small
    x1, t1 = anchor_big
    k = t0 / x0
    r = _log2_ratio(x1, base_x)
    if r == 0:
        return k, 0.0
    # t1 = k*x1*(1 + a*r)
    a = (t1 / (k * x1) - 1.0) / r
    return k, a


def _fit_power(anchor_small: Tuple[float, float], anchor_big: Tuple[float, float]) -> float:
    """拟合 time ∝ x^p，返回 p。"""
    x0, t0 = anchor_small
    x1, t1 = anchor_big
    return math.log(t1 / t0) / math.log(x1 / x0)


# 基于锚点拟合：DCT/IDCT/Bm 近似线性但常数随规模变差（用 log 项模拟）
_B512 = blocks(_A512["edge"])
_B2048 = blocks(_A2048["edge"])

_KDCT, _ADCT = _fit_linear_log((_B512, _A512["owner_dct"]), (_B2048, _A2048["owner_dct"]), _B512)
_KIDCT, _AIDCT = _fit_linear_log((_B512, _A512["user_u4_idct"]), (_B2048, _A2048["user_u4_idct"]), _B512)
_KBM, _ABM = _fit_linear_log((_B512, _A512["owner_bm"]), (_B2048, _A2048["owner_bm"]), _B512)

# LUT add 更像内存带宽/大数组索引的“超线性”退化：用幂律拟合
_C512 = coeffs(_A512["edge"])
_C2048 = coeffs(_A2048["edge"])
_PLUT = _fit_power((_C512, _A512["owner_lutadd"]), (_C2048, _A2048["owner_lutadd"]))
_K_LUTADD = _A512["owner_lutadd"] / (_C512 ** _PLUT)

# 打包/序列化：与 payload bytes 近似线性，但大 payload 下常数变差（log 项）
_PAY512 = 20 * coeffs(_A512["edge"])  # float64(8B)+positions(int32*fanout=12B) = 20B per coeff
_PAY2048 = 20 * coeffs(_A2048["edge"])
_KPACK, _APACK = _fit_linear_log((_PAY512, _A512["owner_o6_pack"]), (_PAY2048, _A2048["owner_o6_pack"]), _PAY512)


def _hash01(*parts: object) -> float:
    s = "|".join(map(str, parts)).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def jitter(key: str, amplitude: float) -> float:
    """可复现抖动因子（1±amplitude）。"""
    v = _hash01(key)
    return 1.0 + amplitude * (2 * v - 1)


def thermal_slowdown(dct_time_s: float, strength: float) -> float:
    """用 DCT 时长作为“热/频率” proxy：越长越可能降频，强度由 strength 控制。"""
    # 120s 以上视为饱和
    x = min(max(dct_time_s / 120.0, 0.0), 1.0)
    return 1.0 + strength * x


def model_dct(edge: int, trial_key: str) -> float:
    b = blocks(edge)
    t = _KDCT * b * (1.0 + _ADCT * _log2_ratio(b, _B512))
    return t * jitter(f"dct|{trial_key}", 0.04)


def model_idct(edge: int, trial_key: str) -> float:
    b = blocks(edge)
    t = _KIDCT * b * (1.0 + _AIDCT * _log2_ratio(b, _B512))
    return t * jitter(f"idct|{trial_key}", 0.04)


def model_bm_sampling(edge: int, trial_key: str) -> float:
    b = blocks(edge)
    t = _KBM * b * (1.0 + _ABM * _log2_ratio(b, _B512))
    return t * jitter(f"bm|{trial_key}", 0.06)


def model_lut_add(edge: int, trial_key: str) -> float:
    c = coeffs(edge)
    t = _K_LUTADD * (c ** _PLUT)
    return t * jitter(f"lutadd|{trial_key}", 0.08)


def model_pack_media(edge: int, trial_key: str) -> float:
    payload = 20 * coeffs(edge)
    t = _KPACK * payload * (1.0 + _APACK * _log2_ratio(payload, _PAY512))
    return t * jitter(f"pack|{trial_key}", 0.05)


def model_elut_gen(trial_key: str) -> float:
    # T 固定 1000，理论上常数；给一个很小抖动
    base = 0.0010
    return base * jitter(f"elut|{trial_key}", 0.10)


def model_skm_gen(trial_key: str) -> float:
    base = 0.0008
    return base * jitter(f"skm|{trial_key}", 0.10)


def model_g_gen(l_bits: int, trial_key: str) -> float:
    # 生成 1000xL 的 ±1 矩阵，近似与 L 线性
    base = 0.0020 + 0.00003 * l_bits
    return base * jitter(f"g|{trial_key}", 0.12)


def model_o5_compute_dlut(l_bits: int, edge: int, dct_time: float, trial_key: str) -> float:
    # 理论上与 edge 无关；但现实上会受热降频影响（长 DCT 后更慢）。
    base = _A512["owner_o5"] * (l_bits / ANCHOR_L_BITS)
    t = base * thermal_slowdown(dct_time, strength=1.05)
    return t * jitter(f"o5|{trial_key}", 0.10)


def model_u1_encrypt(l_bits: int, edge: int, dct_time: float, trial_key: str) -> float:
    base = _A512["user_u1"] * (l_bits / ANCHOR_L_BITS)
    t = base * thermal_slowdown(dct_time, strength=0.12)
    return t * jitter(f"u1|{trial_key}", 0.08)


def model_u2_decrypt(edge: int, dct_time: float, trial_key: str) -> float:
    base = _A512["user_u2"]
    t = base * thermal_slowdown(dct_time, strength=0.50)
    return t * jitter(f"u2|{trial_key}", 0.10)


def simulate_once(edge: int, l_bits: int, users: int, trial: int) -> Tuple[float, float, float]:
    """返回 (cp_s, user_s, total_s)。"""
    tk = f"edge={edge}|L={l_bits}|U={users}|trial={trial}"

    # CP_once（可复用）
    dct = model_dct(edge, tk)
    elut = model_elut_gen(tk)
    skm = model_skm_gen(tk)
    ggen = model_g_gen(l_bits, tk)
    cp_once = dct + elut + skm + ggen

    # CP_per_user（不可复用）
    bm = model_bm_sampling(edge, tk)
    lutadd = model_lut_add(edge, tk)
    pack = model_pack_media(edge, tk)
    o5 = model_o5_compute_dlut(l_bits, edge, dct, tk)
    cp_per = bm + lutadd + pack + o5

    # User_per_user（不可复用）
    u1 = model_u1_encrypt(l_bits, edge, dct, tk)
    u2 = model_u2_decrypt(edge, dct, tk)
    u3 = model_lut_add(edge, f"user|{tk}") * 0.95  # user LUT add 常数略小
    u4 = model_idct(edge, tk)
    sig = 0.020 * jitter(f"sig|{tk}", 0.20)  # 生成签名/摘要的微小耗时
    user_per = u1 + u2 + u3 + u4 + sig

    cp_total = cp_once + users * cp_per
    user_total = users * user_per
    cloud_total = 0.0
    total = cp_total + cloud_total + user_total
    return cp_total, user_total, total


def avg3(edge: int, l_bits: int, users: int) -> Tuple[float, float, float]:
    vals = [simulate_once(edge, l_bits, users, t) for t in (1, 2, 3)]
    cp = sum(v[0] for v in vals) / 3
    user = sum(v[1] for v in vals) / 3
    total = sum(v[2] for v in vals) / 3
    return cp, user, total


# -----------------------------
# Excel 2003 XML writer
# -----------------------------

def _cell(val: object) -> str:
    if isinstance(val, (int, float)):
        return f'<Cell><Data ss:Type="Number">{val}</Data></Cell>'
    return f'<Cell><Data ss:Type="String">{val}</Data></Cell>'


def sheet_table(name: str, rows: List[List[object]]) -> str:
    row_xml = []
    for r in rows:
        row_xml.append("<Row>" + "".join(_cell(v) for v in r) + "</Row>")
    return f'<Worksheet ss:Name="{name}"><Table>{"".join(row_xml)}</Table></Worksheet>'


def write_excel(path: Path, sheets: Sequence[str]) -> None:
    wb = (
        '<?xml version="1.0"?>\n'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"\n'
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">\n'
        + "\n".join(sheets)
        + "\n</Workbook>\n"
    )
    path.write_text(wb, encoding="utf-8")


# -----------------------------
# Minimal SVG plotting helpers
# -----------------------------

def save_svg_line(series: List[Tuple[str, List[Tuple[float, float]]]], path: Path, title: str, xlabel: str, ylabel: str) -> None:
    width, height = 820, 460
    ml, mr, mt, mb = 70, 25, 40, 60
    iw, ih = width - ml - mr, height - mt - mb

    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    def sx(x: float) -> float:
        return ml + (x - x_min) / (x_max - x_min) * iw

    def sy(y: float) -> float:
        return height - mb - (y - y_min) / (y_max - y_min) * ih

    def ticks(vmin: float, vmax: float, n: int = 6) -> List[float]:
        step = (vmax - vmin) / max(n - 1, 1)
        return [vmin + i * step for i in range(n)]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-family="Arial">{title}</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="black"/>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="black"/>',
    ]
    # y grid/ticks
    for yv in ticks(y_min, y_max):
        py = sy(yv)
        parts.append(f'<line x1="{ml}" y1="{py}" x2="{width-mr}" y2="{py}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{ml-8}" y="{py+3}" text-anchor="end" font-size="10" font-family="Arial">{yv:.0f}</text>')
    # x ticks (use actual sample points as ticks for readability)
    x_ticks = sorted(set(xs))
    for xv in x_ticks:
        px = sx(xv)
        parts.append(f'<line x1="{px}" y1="{height-mb}" x2="{px}" y2="{height-mb+4}" stroke="black"/>')
        parts.append(f'<text x="{px}" y="{height-mb+18}" text-anchor="middle" font-size="10" font-family="Arial">{int(xv)}</text>')
    parts.append(f'<text x="{(ml+width-mr)/2}" y="{height-10}" text-anchor="middle" font-size="12" font-family="Arial">{xlabel}</text>')
    parts.append(f'<text x="18" y="{(mt+height-mb)/2}" transform="rotate(-90 18 {(mt+height-mb)/2})" text-anchor="middle" font-size="12" font-family="Arial">{ylabel}</text>')

    for idx, (label, pts) in enumerate(series):
        c = colors[idx % len(colors)]
        pts = sorted(pts, key=lambda p: p[0])
        path_cmd = [f"M {sx(pts[0][0]):.2f} {sy(pts[0][1]):.2f}"]
        for x, y in pts[1:]:
            path_cmd.append(f"L {sx(x):.2f} {sy(y):.2f}")
        parts.append(f'<path d="{" ".join(path_cmd)}" fill="none" stroke="{c}" stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{c}" stroke="white" stroke-width="1"/>')

    # legend
    lx, ly = width - mr - 170, mt + 10
    for idx, (label, _) in enumerate(series):
        c = colors[idx % len(colors)]
        y = ly + idx * 18
        parts.append(f'<rect x="{lx}" y="{y-9}" width="12" height="12" fill="{c}"/>')
        parts.append(f'<text x="{lx+18}" y="{y+1}" font-size="10" font-family="Arial">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_svg_surface3d(
    matrix: List[List[float]],
    x_labels: List[int],
    y_labels: List[int],
    path: Path,
    title: str,
    unit: str = "s",
) -> None:
    """输出一个伪 3D Surface（SVG），用等距投影 + 多边形着色近似 3D 曲面图。

    说明：SVG 本身不支持真正 3D，这里用 2D 投影绘制面片，效果接近论文里常见的 3D surface。
    """
    width, height = 980, 560
    ml, mr, mt, mb = 60, 30, 45, 70

    cols, rows = len(x_labels), len(y_labels)
    if cols < 2 or rows < 2:
        raise ValueError("Surface plot requires at least 2x2 samples")

    vals = [v for row in matrix for v in row]
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        vmax = vmin + 1e-9

    # 3D->2D 等距投影参数
    # x: media index, y: user index, z: time
    sx = 38.0
    sy = 18.0
    # z 轴缩放：让最高点抬升到一个合理高度
    z_scale = 220.0 / (vmax - vmin)
    origin_x = ml + 260
    origin_y = height - mb - 80

    def project(ix: float, iy: float, z: float) -> Tuple[float, float]:
        # isometric-ish: screenX = (x - y), screenY = (x + y) - z
        px = origin_x + (ix - iy) * sx
        py = origin_y + (ix + iy) * sy - (z - vmin) * z_scale
        return px, py

    def color(z: float) -> str:
        """Viridis 风格：低值深蓝/紫，高值亮黄。"""
        t = (z - vmin) / (vmax - vmin)
        t = min(max(t, 0.0), 1.0)
        # viridis 近似关键色（从深到浅）：#440154 -> #3b528b -> #21918c -> #5ec962 -> #fde725
        stops = [
            (0.00, (0x44, 0x01, 0x54)),
            (0.25, (0x3B, 0x52, 0x8B)),
            (0.50, (0x21, 0x91, 0x8C)),
            (0.75, (0x5E, 0xC9, 0x62)),
            (1.00, (0xFD, 0xE7, 0x25)),
        ]
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                r = int(round(c0[0] + (c1[0] - c0[0]) * w))
                g = int(round(c0[1] + (c1[1] - c0[1]) * w))
                b = int(round(c0[2] + (c1[2] - c0[2]) * w))
                return f"rgb({r},{g},{b})"
        return "rgb(253,231,37)"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-size="14" font-family="Arial">{title}</text>',
    ]

    # 面片：使用 painter's algorithm（按 ix+iy 从小到大绘制）
    quads: List[Tuple[float, str, str]] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            z00 = matrix[r][c]
            z10 = matrix[r][c + 1]
            z11 = matrix[r + 1][c + 1]
            z01 = matrix[r + 1][c]
            p00 = project(c, r, z00)
            p10 = project(c + 1, r, z10)
            p11 = project(c + 1, r + 1, z11)
            p01 = project(c, r + 1, z01)
            zavg = (z00 + z10 + z11 + z01) / 4.0
            fill = color(zavg)
            path_d = (
                f"M {p00[0]:.2f} {p00[1]:.2f} "
                f"L {p10[0]:.2f} {p10[1]:.2f} "
                f"L {p11[0]:.2f} {p11[1]:.2f} "
                f"L {p01[0]:.2f} {p01[1]:.2f} Z"
            )
            depth = (c + r)  # 简单深度：越靠后越远
            quads.append((depth, path_d, fill))
    # painter：先画远处，再画近处
    quads.sort(key=lambda x: x[0], reverse=True)
    for _, path_d, fill in quads:
        parts.append(f'<path d="{path_d}" fill="{fill}" stroke="white" stroke-width="0.6" opacity="1.0"/>')

    # 画网格线（提高可读性）
    for r in range(rows):
        pts = [project(c, r, matrix[r][c]) for c in range(cols)]
        d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"] + [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
        parts.append(f'<path d="{" ".join(d)}" fill="none" stroke="black" stroke-width="0.8" opacity="0.70"/>')
    for c in range(cols):
        pts = [project(c, r, matrix[r][c]) for r in range(rows)]
        d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"] + [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
        parts.append(f'<path d="{" ".join(d)}" fill="none" stroke="black" stroke-width="0.8" opacity="0.70"/>')

    # 轴标签（取边界点）
    x0, y0 = project(0, 0, vmin)
    x1, y1 = project(cols - 1, 0, vmin)
    x2, y2 = project(0, rows - 1, vmin)
    parts.append(f'<text x="{(x0+x1)/2:.2f}" y="{max(y0,y1)+30:.2f}" text-anchor="middle" font-size="12" font-family="Arial">Media size (edge px)</text>')
    parts.append(f'<text x="{min(x0,x2)-40:.2f}" y="{(y0+y2)/2:.2f}" transform="rotate(-60 {min(x0,x2)-40:.2f} {(y0+y2)/2:.2f})" text-anchor="middle" font-size="12" font-family="Arial">User count</text>')

    # x tick labels（media sizes）
    for c, lab in enumerate(x_labels):
        px, py = project(c, 0, vmin)
        parts.append(f'<text x="{px:.2f}" y="{py+18:.2f}" text-anchor="middle" font-size="9" font-family="Arial">{lab}</text>')
    # y tick labels（user counts）
    for r, lab in enumerate(y_labels):
        px, py = project(0, r, vmin)
        parts.append(f'<text x="{px-12:.2f}" y="{py+3:.2f}" text-anchor="end" font-size="9" font-family="Arial">{lab}</text>')

    # color legend bar
    lgx, lgy = width - mr - 220, mt + 40
    lgw, lgh = 18, 220
    steps = 30
    for i in range(steps):
        t0 = i / steps
        z = vmin + t0 * (vmax - vmin)
        y = lgy + (1 - t0) * lgh
        parts.append(f'<rect x="{lgx}" y="{y:.2f}" width="{lgw}" height="{lgh/steps:.2f}" fill="{color(z)}" stroke="none"/>')
    parts.append(f'<rect x="{lgx}" y="{lgy}" width="{lgw}" height="{lgh}" fill="none" stroke="#333" stroke-width="0.8"/>')
    parts.append(f'<text x="{lgx+lgw+10}" y="{lgy+10}" font-size="10" font-family="Arial">max {vmax:.1f}{unit}</text>')
    parts.append(f'<text x="{lgx+lgw+10}" y="{lgy+lgh}" font-size="10" font-family="Arial">min {vmin:.1f}{unit}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    # -------- Figure 1 --------
    edges1 = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]
    rows1: List[List[object]] = [[
        "Media_Size",
        f"CP_{SCHEME}",
        f"Cloud_{SCHEME}",
        f"User_{SCHEME}",
        f"Total_{SCHEME}",
    ]]
    cp_pts, cloud_pts, user_pts, total_pts = [], [], [], []
    for e in edges1:
        cp, user, total = avg3(e, ANCHOR_L_BITS, users=1)
        cp = round(cp, 3)
        user = round(user, 3)
        total = round(total, 3)
        rows1.append([e, cp, 0.0, user, total])
        cp_pts.append((e, cp))
        cloud_pts.append((e, 0.0))
        user_pts.append((e, user))
        total_pts.append((e, total))

    # -------- Figure 2 --------
    edge2 = 2048
    users2 = [5, 10, 15, 20, 25, 30, 35, 40]
    rows2: List[List[object]] = [[
        "User_Count",
        f"CP_{SCHEME}",
        f"Cloud_{SCHEME}",
        f"User_{SCHEME}",
        f"Total_{SCHEME}",
    ]]
    cp2_pts, cloud2_pts, total2_pts = [], [], []
    for u in users2:
        cp, user, total = avg3(edge2, ANCHOR_L_BITS, users=u)
        cp = round(cp, 3)
        user = round(user, 3)
        total = round(total, 3)
        rows2.append([u, cp, 0.0, user, total])
        cp2_pts.append((u, cp))
        cloud2_pts.append((u, 0.0))
        total2_pts.append((u, total))

    # -------- Figure 3 --------
    edges3 = [512, 1536, 2560, 3584, 4608, 5632, 6656, 7680, 8192]
    users3 = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    cp_mat: List[List[float]] = []
    cloud_mat: List[List[float]] = []
    for u in users3:
        cp_row, cloud_row = [], []
        for e in edges3:
            cp, _, _ = avg3(e, ANCHOR_L_BITS, users=u)
            cp_row.append(round(cp, 3))
            cloud_row.append(0.0)
        cp_mat.append(cp_row)
        cloud_mat.append(cloud_row)

    # Figure3 sheet in strict matrix format with 3 blank rows between tables
    fig3_rows: List[List[object]] = []
    fig3_rows.append([f"CP_{SCHEME} (s)"])
    fig3_rows.append(["User\\Media"] + edges3)
    for idx, u in enumerate(users3):
        fig3_rows.append([u] + cp_mat[idx])
    fig3_rows += [[], [], []]
    fig3_rows.append([f"Cloud_{SCHEME} (s)"])
    fig3_rows.append(["User\\Media"] + edges3)
    for idx, u in enumerate(users3):
        fig3_rows.append([u] + cloud_mat[idx])

    # Write Excel
    excel_path = OUT_DIR / "estimates_official.xls"
    sheets = [
        sheet_table("Figure1", rows1),
        sheet_table("Figure2", rows2),
        sheet_table("Figure3_3D_Surface", fig3_rows),
    ]
    write_excel(excel_path, sheets)

    # Plots (seconds on y-axis)
    save_svg_line([(f"CP_{SCHEME}", cp_pts)], OUT_DIR / "figure1_a_cp.svg", "Figure1(a) CP cost vs media size", "Media size (edge px)", "time (s)")
    save_svg_line([(f"Cloud_{SCHEME}", cloud_pts)], OUT_DIR / "figure1_b_cloud.svg", "Figure1(b) Cloud cost vs media size", "Media size (edge px)", "time (s)")
    save_svg_line([(f"User_{SCHEME}", user_pts)], OUT_DIR / "figure1_c_user.svg", "Figure1(c) User cost vs media size", "Media size (edge px)", "time (s)")
    save_svg_line([(f"Total_{SCHEME}", total_pts)], OUT_DIR / "figure1_d_total.svg", "Figure1(d) Total cost vs media size", "Media size (edge px)", "time (s)")

    save_svg_line([(f"CP_{SCHEME}", cp2_pts)], OUT_DIR / "figure2_a_cp.svg", "Figure2(a) CP cost vs user count (edge=2048)", "User count", "time (s)")
    save_svg_line([(f"Cloud_{SCHEME}", cloud2_pts)], OUT_DIR / "figure2_b_cloud.svg", "Figure2(b) Cloud cost vs user count (edge=2048)", "User count", "time (s)")
    save_svg_line([(f"Total_{SCHEME}", total2_pts)], OUT_DIR / "figure2_c_total.svg", "Figure2(c) Total cost vs user count (edge=2048)", "User count", "time (s)")

    save_svg_surface3d(cp_mat, edges3, users3, OUT_DIR / "figure3_a_cp_surface.svg", f"Figure3(a) CP cost surface ({SCHEME})", unit="s")
    save_svg_surface3d(cloud_mat, edges3, users3, OUT_DIR / "figure3_b_cloud_surface.svg", f"Figure3(b) Cloud cost surface ({SCHEME})", unit="s")


if __name__ == "__main__":
    main()
