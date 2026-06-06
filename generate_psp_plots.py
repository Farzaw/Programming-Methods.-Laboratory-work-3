#!/usr/bin/env python3
"""
@file generate_psp_plots.py
@brief Построение графиков для лабораторной работы по ПСП.

Читает CSV-файлы из output/ и сохраняет PNG-графики в output/plots/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
PLOTS = OUTPUT / "plots"

# Цветовая схема генераторов
COLORS = {
    "LCG": "#2196F3",
    "VonNeumann+Weyl": "#FF9800",
    "Xorshift32": "#4CAF50",
    "std_mt19937": "#9C27B0",
}

GENERATOR_ORDER = ["LCG", "VonNeumann+Weyl", "Xorshift32", "std_mt19937"]


def setup_style() -> None:
    """Настраивает стиль matplotlib для русских подписей."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
    })


def load_csv(name: str) -> pd.DataFrame:
    """Загружает CSV из output/ с проверкой существования."""
    path = OUTPUT / name
    if not path.exists():
        print(f"Ошибка: файл не найден — {path}")
        print("Сначала запустите: psp_lab.exe")
        sys.exit(1)
    return pd.read_csv(path)


def plot_speed_benchmark(df: pd.DataFrame) -> None:
    """График 1: сравнение времени генерации."""
    fig, ax = plt.subplots(figsize=(10, 6))

    columns = ["LCG", "VonNeumann", "Xorshift", "std_mt19937"]
    labels = ["LCG", "VonNeumann+Weyl", "Xorshift32", "std::mt19937"]
    markers = ["o", "s", "^", "D"]

    sizes = df["size"].values
    for col, label, marker in zip(columns, labels, markers):
        color = COLORS.get(label, COLORS.get(col, "#333333"))
        ax.plot(sizes, df[col], marker=marker, linewidth=2, markersize=7,
                label=label, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Размер выборки (элементов)")
    ax.set_ylabel("Время генерации (мс)")
    ax.set_title("Бенчмарк производительности генераторов ПСП")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Подписи размеров на оси X
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s // 1000}k" if s >= 1000 else str(s) for s in sizes])

    fig.tight_layout()
    out = PLOTS / "speed_benchmark.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}")


def plot_chi2_pass_rate(df: pd.DataFrame) -> None:
    """График 2: доля выборок, прошедших критерий Хи-квадрат."""
    grouped = df.groupby("generator")["chi2_passed"].agg(["sum", "count"]).reset_index()
    grouped["pass_rate"] = grouped["sum"] / grouped["count"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    gens = [g for g in GENERATOR_ORDER[:3] if g in grouped["generator"].values]
    rates = [grouped.loc[grouped["generator"] == g, "pass_rate"].values[0] for g in gens]
    colors = [COLORS[g] for g in gens]

    bars = ax.bar(gens, rates, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.6, label="95% (эталон)")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Доля принятых выборок (%)")
    ax.set_title("Критерий Хи-квадрат: гипотеза о равномерности (α = 0.05)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = PLOTS / "chi2_pass_rate.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}")


def plot_nist_results(df: pd.DataFrame) -> None:
    """График 3: p-value NIST-тестов по генераторам."""
    generators = [g for g in GENERATOR_ORDER[:3] if g in df["generator"].unique()]
    tests = df["test"].unique().tolist()
    n_gen = len(generators)
    n_test = len(tests)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(n_test)
    width = 0.25

    for i, gen in enumerate(generators):
        subset = df[df["generator"] == gen].set_index("test").loc[tests]
        pvals = subset["p_value"].clip(0, 1).values
        offset = (i - n_gen / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], pvals, width,
                      label=gen, color=COLORS[gen], alpha=0.85)

        for bar, pval, passed in zip(bars, pvals, subset["passed"].values):
            marker = "✓" if passed else "✗"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    marker, ha="center", va="bottom", fontsize=9)

    ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1.5,
               label="α = 0.05 (порог)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace(" Test", "").replace(" within Block", "\nwithin Block")
                        for t in tests], rotation=15, ha="right")
    ax.set_ylabel("p-value")
    ax.set_title("NIST/Diehard тесты: p-value по генераторам (✓ — принято, ✗ — отклонено)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = PLOTS / "nist_results.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}")


def plot_histogram_uniformity(df: pd.DataFrame) -> None:
    """График 4: гистограммы равномерности (10 карманов)."""
    generators = [g for g in GENERATOR_ORDER[:3] if g in df["generator"].unique()]
    expected = 1000 / 10  # 1000 элементов / 10 карманов

    fig, axes = plt.subplots(1, len(generators), figsize=(14, 5), sharey=True)
    if len(generators) == 1:
        axes = [axes]

    for ax, gen in zip(axes, generators):
        subset = df[df["generator"] == gen].sort_values("bin")
        bins = [f"{int(r.bin_start)}–{int(r.bin_end)}"
                for r in subset.itertuples()]
        counts = subset["count"].values

        ax.bar(range(len(counts)), counts, color=COLORS[gen], alpha=0.8,
               edgecolor="white")
        ax.axhline(y=expected, color="red", linestyle="--", linewidth=1.5,
                   label=f"Ожидаемое ({expected:.0f})")
        ax.set_title(gen)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Карман (диапазон)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Частота")
    fig.suptitle("Гистограммы распределения (выборка 1, n = 1000, K = 10 карманов)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = PLOTS / "histogram_uniformity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}")


def plot_stats_boxplot(df: pd.DataFrame) -> None:
    """График 5: boxplot среднего и СКО по 20 выборкам."""
    generators = [g for g in GENERATOR_ORDER[:3] if g in df["generator"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes, ["mean", "std_dev"],
                                  ["Среднее значение", "СКО"]):
        data = [df.loc[df["generator"] == g, metric].values for g in generators]
        bp = ax.boxplot(data, tick_labels=generators, patch_artist=True)

        for patch, gen in zip(bp["boxes"], generators):
            patch.set_facecolor(COLORS[gen])
            patch.set_alpha(0.7)

        theoretical_mean = (5000 + 0) / 2
        if metric == "mean":
            ax.axhline(y=theoretical_mean, color="red", linestyle="--",
                       label=f"Теор. среднее ({theoretical_mean})")
            ax.legend()

        ax.set_title(title)
        ax.set_ylabel("Значение")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Разброс описательной статистики (20 выборок на генератор)", fontsize=13)
    fig.tight_layout()
    out = PLOTS / "stats_boxplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}")


def create_html_report() -> None:
    """Создаёт HTML-отчёт со встроенными графиками."""
    plots = [
        ("speed_benchmark.png", "Бенчмарк производительности"),
        ("chi2_pass_rate.png", "Критерий Хи-квадрат"),
        ("nist_results.png", "NIST-тесты"),
        ("histogram_uniformity.png", "Гистограммы равномерности"),
        ("stats_boxplot.png", "Описательная статистика"),
    ]

    sections = ""
    for filename, title in plots:
        sections += f"""
        <section>
            <h2>{title}</h2>
            <img src="plots/{filename}" alt="{title}">
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Отчёт: Генерация и тестирование ПСП</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1100px; margin: 0 auto;
                padding: 24px; background: #f5f5f5; color: #222; }}
        h1 {{ color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 8px; }}
        h2 {{ color: #333; margin-top: 32px; }}
        section {{ background: #fff; border-radius: 8px; padding: 20px;
                    margin: 16px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .meta {{ color: #666; font-size: 0.95em; }}
        table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #E3F2FD; }}
    </style>
</head>
<body>
    <h1>Лабораторная работа: Генерация и тестирование ПСП</h1>
    <p class="meta">Диапазон [0, 5000] · 20 выборок × 1000 элементов · α = 0.05</p>

    <section>
        <h2>Генераторы</h2>
        <table>
            <tr><th>Генератор</th><th>Формула / метод</th></tr>
            <tr><td>LCG</td><td>X<sub>n+1</sub> = (1103515245·X<sub>n</sub> + 12345) mod 2³¹</td></tr>
            <tr><td>Von Neumann+Weyl</td><td>middle(X² + w), w += 0x9E3779B97F4A7C15</td></tr>
            <tr><td>Xorshift32</td><td>x ^= x&lt;&lt;13; x ^= x&gt;&gt;17; x ^= x&lt;&lt;5</td></tr>
        </table>
    </section>
{sections}
    <section>
        <h2>Файлы данных</h2>
        <ul>
            <li><code>output/speed_results.csv</code> — бенчмарк</li>
            <li><code>output/stats_summary.csv</code> — статистика выборок</li>
            <li><code>output/nist_results.csv</code> — NIST-тесты</li>
            <li><code>output/histogram.csv</code> — гистограммы</li>
        </ul>
    </section>
</body>
</html>"""

    report_path = OUTPUT / "psp_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  + {report_path.name}")


def main() -> None:
    """Главная функция: строит все графики и HTML-отчёт."""
    print("Построение графиков для лабораторной работы по ПСП...")
    PLOTS.mkdir(parents=True, exist_ok=True)
    setup_style()

    speed_df = load_csv("speed_results.csv")
    stats_df = load_csv("stats_summary.csv")
    nist_df = load_csv("nist_results.csv")
    hist_df = load_csv("histogram.csv")

    plot_speed_benchmark(speed_df)
    plot_chi2_pass_rate(stats_df)
    plot_nist_results(nist_df)
    plot_histogram_uniformity(hist_df)
    plot_stats_boxplot(stats_df)
    create_html_report()

    print(f"\nГотово! Графики: {PLOTS}")
    print(f"HTML-отчёт: {OUTPUT / 'psp_report.html'}")


if __name__ == "__main__":
    main()
