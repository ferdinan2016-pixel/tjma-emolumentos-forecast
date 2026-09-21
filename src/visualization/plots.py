"""
plots.py

Geração dos gráficos comparativos exigidos no Capítulo 4 do Projeto de
Qualificação: (i) visão geral da série histórica sintética; (ii) previsão
um-passo-à-frente das três arquiteturas contra o valor real, no período de
teste; (iii) comparação de métricas de erro; (iv) previsão multi-step
("ano seguinte") das três arquiteturas.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

CORES = {"ARIMA/SARIMA": "#1f6f4d", "LSTM": "#b3401f", "XGBoost": "#2a4a9e", "Real": "#1a1a1a"}


def _fmt_reais(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:,.1f} mi"))


def plot_serie_geral(df: pd.DataFrame, split_date, caminho_saida: str):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["arrecadacao"], color="#1a1a1a", linewidth=1.3)
    ax.axvline(split_date, color="gray", linestyle="--", linewidth=1)
    ax.text(split_date, df["arrecadacao"].max() * 0.97, "  início do teste (2025)", fontsize=9, color="gray")
    ax.set_title("Série sintética de arrecadação de emolumentos extrajudiciais — TJMA (2012–2025)\n"
                  "(placeholder até a obtenção da série real junto ao FERJ/Corregedoria)")
    _fmt_reais(ax)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plot_comparacao_teste(datas_teste, y_real, previsoes: dict, caminho_saida: str):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(datas_teste, y_real, color=CORES["Real"], marker="o", label="Real", linewidth=2)
    for nome, y_pred in previsoes.items():
        ax.plot(datas_teste, y_pred, marker="o", linestyle="--", label=nome, color=CORES.get(nome))
    ax.set_title("Previsão um-passo-à-frente (walk-forward) no período de teste (2025)")
    _fmt_reais(ax)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plot_metricas(df_metricas: pd.DataFrame, caminho_saida: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, coluna in zip(axes, ["MAE", "RMSE", "MAPE (%)"]):
        cores = [CORES.get(m, "#888") for m in df_metricas.index]
        ax.bar(df_metricas.index, df_metricas[coluna], color=cores)
        ax.set_title(coluna)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Comparação de métricas de erro entre arquiteturas (conjunto de teste, 2025)")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plot_forecast_proximo_ano(df_historico: pd.DataFrame, datas_futuras, previsoes: dict, caminho_saida: str):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    historico_recente = df_historico["arrecadacao"].iloc[-24:]
    ax.plot(historico_recente.index, historico_recente.values, color=CORES["Real"], label="Histórico (real/sintético)")
    for nome, y_pred in previsoes.items():
        ax.plot(datas_futuras, y_pred, marker="o", linestyle="--", label=f"Previsão {nome}", color=CORES.get(nome))
    ax.axvline(df_historico.index[-1], color="gray", linestyle=":", linewidth=1)
    ax.set_title("Previsão da arrecadação do ano seguinte (2026) — horizonte de 12 meses")
    _fmt_reais(ax)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
