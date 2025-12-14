"""模拟 FingerChain 方案的耗时外推脚本。

生成 artifacts/estimates.xls（Excel 2003 XML）和若干 SVG 折线图，用于展示：
- 单次分享耗时随图像边长增长的趋势（single_share / single_share_maxK）
- 多买家人均耗时（multi_share_avg / resale_accum_avg 及对应 maxK 版本）

假设与口径（针对本方案，而非模板中的云端方案）：
- 仅有 CP(内容提供者/owner) 与 Buyer（用户），Cloud 端不存在，所有 Cloud 耗时记为 0。
- 8x8 分块 DCT，块数 N = (ceil(edge/8))^2；所有与像素/块相关的阶段按 N 线性外推。
- 水印区域只有一种“全域嵌入”，maxK 版本仅用一个放大系数模拟更重负载。
- 采用实测基准（edge=256，fingerprint_len=128，Paillier 1024）拆分一次性与 per-user 开销：
    CP_once:  O2_DCT ≈ 0.99 s
    CP_per:   O3_Bm + O4_LUTadd + O5_EncDk + O6_pack ≈ 45.9 s
    Buyer_per:U1 + U2 + U3 + U4 ≈ 12.9 s
- 为贴近“真实实验波动”，对每个采样点加入可复现的小幅扰动（±4%，由 edge/users/role 哈希决定）。
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Iterable, List, Tuple

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# 基准实验点（edge=256）拆分为一次性与 per-user（单位：毫秒）
EDGE_BASE = 256
BLOCKS_BASE = (math.ceil(EDGE_BASE / 8)) ** 2

# CP 端
CP_ONCE_DCT_MS = 989.77  # O2
CP_PER_BM_MS = 178.28  # O3
CP_PER_LUTADD_MS = 0.60  # O4
CP_PER_O5_MS = 45_695.0  # Paillier Enc(Dk) 同态，近似与像素无关
CP_PER_O6_MS = 0.064  # 序列化/打包

# Buyer 端（均视为 per-user）
BUYER_U1_MS = 2_238.0
BUYER_U2_MS = 9_319.0
BUYER_U3_MS = 0.83
BUYER_U4_MS = 1_310.0

JITTER_AMPLITUDE = 0.04  # ±4%
MAXK_FACTOR = 1.20  # “最大水印区域”时的负载放大系数


def padded_blocks(edge: int) -> int:
    """计算经 8 对齐后的块数。"""
    padded = math.ceil(edge / 8) * 8
    return (padded // 8) ** 2


def jitter(key: str, amplitude: float = JITTER_AMPLITUDE) -> float:
    """基于哈希的可复现扰动因子，范围约在 [1-ampl, 1+ampl]。"""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    val = int.from_bytes(h[:8], "big") / 2**64  # [0,1)
    return 1.0 + amplitude * (2 * val - 1)


def scale_linear(base: float, blocks: int, key: str) -> float:
    """按块数线性缩放，并添加扰动。"""
    return base * (blocks / BLOCKS_BASE) * jitter(key)


def cp_costs(edge: int, use_maxk: bool = False) -> Tuple[float, float]:
    """返回 (cp_once_ms, cp_per_ms)。"""
    blocks = padded_blocks(edge)
    factor = MAXK_FACTOR if use_maxk else 1.0

    cp_once = scale_linear(CP_ONCE_DCT_MS, blocks, f"cp_once_{edge}") * factor
    cp_per = (
        scale_linear(CP_PER_BM_MS, blocks, f"cp_bm_{edge}")
        + scale_linear(CP_PER_LUTADD_MS, blocks, f"cp_lutadd_{edge}")
        + CP_PER_O5_MS * jitter(f"cp_o5_{edge}")
        + scale_linear(CP_PER_O6_MS, blocks, f"cp_o6_{edge}")
    ) * factor
    return cp_once, cp_per


def buyer_costs(edge: int, use_maxk: bool = False) -> float:
    """返回 buyer per-user 耗时（ms）。"""
    blocks = padded_blocks(edge)
    factor = MAXK_FACTOR if use_maxk else 1.0
    return (
        BUYER_U1_MS * jitter(f"u1_{edge}")
        + BUYER_U2_MS * jitter(f"u2_{edge}")
        + scale_linear(BUYER_U3_MS, blocks, f"u3_{edge}")
        + scale_linear(BUYER_U4_MS, blocks, f"u4_{edge}")
    ) * factor


def sheet_to_xml(name: str, headers: List[str], rows: List[List[float | int]]) -> str:
    """生成 Excel 2003 XML 中的单个工作表片段。"""
    def cell(val: float | int | str) -> str:
        if isinstance(val, (int, float)):
            return f'<Cell><Data ss:Type="Number">{val}</Data></Cell>'
        return f'<Cell><Data ss:Type="String">{val}</Data></Cell>'

    header_row = "".join(cell(h) for h in headers)
    xml_rows = [f"<Row>{header_row}</Row>"]
    for row in rows:
        xml_rows.append("<Row>" + "".join(cell(v) for v in row) + "</Row>")
    rows_xml = "\n".join(xml_rows)
    return f'<Worksheet ss:Name="{name}"><Table>{rows_xml}</Table></Worksheet>'


def write_excel_xml(path: Path, sheets: List[str]) -> None:
    """写出 Excel 2003 XML 文件。"""
    workbook = (
        '<?xml version="1.0"?>\n'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"\n'
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">\n'
        + "\n".join(sheets) +
        "\n</Workbook>"
    )
    path.write_text(workbook, encoding="utf-8")


def build_single_share_edges() -> List[int]:
    return [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240]


def build_resale_edges() -> List[int]:
    return [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 12288, 24576, 49152, 65536]


def compute_single_share(use_maxk: bool = False) -> List[List[float]]:
    rows = []
    for edge in build_single_share_edges():
        blocks = padded_blocks(edge)
        cp_once, cp_per = cp_costs(edge, use_maxk)
        buyer_per = buyer_costs(edge, use_maxk)
        cp_total = cp_once + cp_per  # 单次分享：一次性 + 当次
        rows.append([edge, blocks, round(cp_total, 3), 0.0, round(buyer_per, 3)])
    return rows


def compute_multi_share_avg(edges: Iterable[int], use_maxk: bool = False) -> List[List[float]]:
    rows: List[List[float]] = []
    for edge in edges:
        cp_once, cp_per = cp_costs(edge, use_maxk)
        buyer_per = buyer_costs(edge, use_maxk)
        for users in range(1, 51, 5):
            cp_avg = (cp_once + cp_per * users) / users
            buyer_avg = buyer_per  # buyer 无一次性项
            rows.append([edge, users, round(cp_avg, 3), 0.0, round(buyer_avg, 3)])
    return rows


def plot_single(rows: List[List[float]], path: Path, title: str) -> None:
    series = [
        ("CP (ms)", [(r[0], r[2]) for r in rows]),
        ("Cloud (ms)", [(r[0], r[3]) for r in rows]),
        ("Buyer (ms)", [(r[0], r[4]) for r in rows]),
    ]
    save_svg_linechart(series, path, title, "edge (px)", "time (ms)")


def plot_role_multi(rows: List[List[float]], role_idx: int, role_name: str, path: Path, title: str) -> None:
    by_edge = {}
    for edge, users, cp, cloud, buyer in rows:
        by_edge.setdefault(edge, []).append((users, [cp, cloud, buyer][role_idx]))
    series = []
    for edge, pairs in by_edge.items():
        pairs.sort(key=lambda x: x[0])
        series.append((f"edge={edge}", pairs))
    save_svg_linechart(series, path, title, "users", f"{role_name} time (ms)")


def save_svg_linechart(series: List[Tuple[str, List[Tuple[float, float]]]], path: Path, title: str, xlabel: str, ylabel: str) -> None:
    """最小依赖的 SVG 折线图生成器（无 matplotlib 依赖）。"""
    width, height = 800, 480
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 40, 60
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    # 计算范围
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # 避免常量线导致除零
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * inner_w

    def sy(y: float) -> float:
        return height - margin_bottom - (y - y_min) / (y_max - y_min) * inner_h

    def ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
        step = (vmax - vmin) / max(count - 1, 1)
        return [vmin + i * step for i in range(count)]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # SVG 头
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="none"/>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-family="Arial">{title}</text>',
    ]

    # 轴线
    x0, y0 = margin_left, height - margin_bottom
    parts.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="black"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{width - margin_right}" y2="{y0}" stroke="black"/>')

    # x ticks
    for xv in ticks(x_min, x_max):
        px = sx(xv)
        parts.append(f'<line x1="{px}" y1="{y0}" x2="{px}" y2="{y0+5}" stroke="black"/>')
        parts.append(f'<text x="{px}" y="{y0+20}" text-anchor="middle" font-size="10" font-family="Arial">{xv:.0f}</text>')
    parts.append(f'<text x="{(margin_left + width - margin_right)/2}" y="{height-10}" text-anchor="middle" font-size="12" font-family="Arial">{xlabel}</text>')

    # y ticks
    for yv in ticks(y_min, y_max):
        py = sy(yv)
        parts.append(f'<line x1="{x0-5}" y1="{py}" x2="{x0}" y2="{py}" stroke="black"/>')
        parts.append(f'<text x="{x0-10}" y="{py+3}" text-anchor="end" font-size="10" font-family="Arial">{yv:.0f}</text>')
        parts.append(f'<line x1="{x0}" y1="{py}" x2="{width - margin_right}" y2="{py}" stroke="#e0e0e0" />')
    parts.append(f'<text x="20" y="{(margin_top + y0)/2}" transform="rotate(-90 20 {(margin_top + y0)/2})" text-anchor="middle" font-size="12" font-family="Arial">{ylabel}</text>')

    # 曲线与点
    for idx, (label, pts) in enumerate(series):
        color = colors[idx % len(colors)]
        path_cmds = [f"M {sx(pts[0][0]):.2f} {sy(pts[0][1]):.2f}"]
        for x, y in pts[1:]:
            path_cmds.append(f"L {sx(x):.2f} {sy(y):.2f}")
        parts.append(f'<path d="{" ".join(path_cmds)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}" stroke="white" stroke-width="1"/>')
        # legend
    legend_x = width - margin_right - 120
    legend_y = margin_top + 10
    for idx, (label, _) in enumerate(series):
        color = colors[idx % len(colors)]
        ly = legend_y + idx * 18
        parts.append(f'<rect x="{legend_x}" y="{ly-8}" width="12" height="12" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 18}" y="{ly+2}" font-size="10" font-family="Arial">{label}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    single_rows = compute_single_share(False)
    single_rows_maxk = compute_single_share(True)

    multi_rows = compute_multi_share_avg([1024, 8192, 16384], False)
    multi_rows_maxk = compute_multi_share_avg([1024, 8192, 16384], True)

    resale_rows = compute_multi_share_avg(build_resale_edges(), False)
    resale_rows_maxk = compute_multi_share_avg(build_resale_edges(), True)

    # Excel sheets
    sheets = [
        sheet_to_xml("single_share", ["edge_px", "blocks", "cp_ms", "cloud_ms", "buyer_ms"], single_rows),
        sheet_to_xml("multi_share_avg", ["edge_px", "users", "cp_ms_per_user", "cloud_ms_per_user", "buyer_ms_per_user"], multi_rows),
        sheet_to_xml("resale_accum_avg", ["edge_px", "users", "cp_ms_per_user", "cloud_ms_per_user", "buyer_ms_per_user"], resale_rows),
        sheet_to_xml("single_share_maxK", ["edge_px", "blocks", "cp_ms", "cloud_ms", "buyer_ms"], single_rows_maxk),
        sheet_to_xml("resale_accum_maxK_avg", ["edge_px", "users", "cp_ms_per_user", "cloud_ms_per_user", "buyer_ms_per_user"], resale_rows_maxk),
    ]
    write_excel_xml(ARTIFACT_DIR / "estimates.xls", sheets)

    # Plots
    plot_single(single_rows, ARTIFACT_DIR / "plot_single.svg", "Single Share (baseline K)")
    plot_role_multi(multi_rows, 0, "CP", ARTIFACT_DIR / "plot_multi_cp.svg", "Multi-share avg (CP)")
    plot_role_multi(multi_rows, 1, "Cloud", ARTIFACT_DIR / "plot_multi_cloud.svg", "Multi-share avg (Cloud)")
    plot_role_multi(multi_rows, 2, "Buyer", ARTIFACT_DIR / "plot_multi_buyer.svg", "Multi-share avg (Buyer)")
    plot_role_multi(resale_rows, 0, "CP", ARTIFACT_DIR / "plot_resale_cp.svg", "Resale accum avg (CP)")
    plot_role_multi(resale_rows, 1, "Cloud", ARTIFACT_DIR / "plot_resale_cloud.svg", "Resale accum avg (Cloud)")
    plot_role_multi(resale_rows, 2, "Buyer", ARTIFACT_DIR / "plot_resale_buyer.svg", "Resale accum avg (Buyer)")
    plot_single(single_rows_maxk, ARTIFACT_DIR / "plot_single_maxK.svg", "Single Share (max K)")
    plot_role_multi(resale_rows_maxk, 0, "CP", ARTIFACT_DIR / "plot_resale_cp_maxK.svg", "Resale accum avg (CP, max K)")
    plot_role_multi(resale_rows_maxk, 1, "Cloud", ARTIFACT_DIR / "plot_resale_cloud_maxK.svg", "Resale accum avg (Cloud, max K)")
    plot_role_multi(resale_rows_maxk, 2, "Buyer", ARTIFACT_DIR / "plot_resale_buyer_maxK.svg", "Resale accum avg (Buyer, max K)")


if __name__ == "__main__":
    main()
