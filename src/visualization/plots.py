"""
plots.py

Geração dos gráficos comparativos exigidos no Capítulo 4 do Projeto de
Qualificação: (i) visão geral da série histórica; (ii) previsão
um-passo-à-frente das sete arquiteturas contra o valor real, no período de
teste; (iii) comparação de métricas de erro; (iv) previsão multi-step
("ano seguinte") das sete arquiteturas; (v) série histórica completa com as
previsões de todas as arquiteturas sobrepostas (teste + previsão futura), para
avaliar visualmente a proximidade entre previsão e valor real ao longo de
todo o horizonte, não apenas na janela de teste.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

CORES = {
    "ARIMA/SARIMA": "#1f6f4d",
    "LSTM": "#b3401f",
    "GRU": "#d98c21",
    "XGBoost": "#2a4a9e",
    "Random Forest": "#6a3d9a",
    "LightGBM": "#1f9e9e",
    "CatBoost": "#c2185b",
    "Real": "#1a1a1a",
}


def _fmt_reais(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:,.1f} mi"))


def plot_serie_geral(df: pd.DataFrame, split_date, caminho_saida: str, fonte: str = "real"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["arrecadacao"], color="#1a1a1a", linewidth=1.3)
    ax.axvline(split_date, color="gray", linestyle="--", linewidth=1)
    ax.text(split_date, df["arrecadacao"].max() * 0.97, "  início do teste", fontsize=9, color="gray")
    ano_ini, ano_fim = df.index[0].year, df.index[-1].year
    if fonte == "real":
        ax.set_title(f"Arrecadação bruta mensal REAL das serventias extrajudiciais — TJMA/FERJ ({ano_ini}–{ano_fim})\n"
                      "(consolidada a partir das publicações mensais oficiais do FERJ)")
    else:
        ax.set_title(f"Série sintética de arrecadação de emolumentos extrajudiciais — TJMA ({ano_ini}–{ano_fim})\n"
                      "(placeholder — ver série real em outputs/figures/01_serie_geral.png)")
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
    periodo = f"{datas_teste[0].strftime('%m/%Y')} a {datas_teste[-1].strftime('%m/%Y')}"
    ax.set_title(f"Previsão um-passo-à-frente (walk-forward) no período de teste ({periodo})")
    _fmt_reais(ax)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plot_metricas(df_metricas: pd.DataFrame, caminho_saida: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, coluna in zip(axes, ["MAE", "RMSE", "MAPE (%)"]):
        cores = [CORES.get(m, "#888") for m in df_metricas.index]
        ax.bar(df_metricas.index, df_metricas[coluna], color=cores)
        ax.set_title(coluna)
        ax.tick_params(axis="x", rotation=45, labelsize=8.5)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Comparação de métricas de erro entre arquiteturas (conjunto de teste)")
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
    periodo = f"{datas_futuras[0].strftime('%m/%Y')} a {datas_futuras[-1].strftime('%m/%Y')}"
    ax.set_title(f"Previsão da arrecadação dos 12 meses seguintes ({periodo})")
    _fmt_reais(ax)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plot_serie_completa_com_previsoes(
    df_historico: pd.DataFrame,
    datas_teste,
    previsoes_teste: dict,
    datas_futuras,
    previsoes_futuras: dict,
    caminho_saida: str,
):
    """Sobrepõe, na série histórica completa, as previsões de cada
    arquitetura no período de teste (walk-forward) e na previsão multi-step
    futura, em um único gráfico contínuo. Permite avaliar visualmente a
    proximidade entre previsão e valor real ao longo de um horizonte maior,
    e não apenas na janela estreita de 12 meses de teste."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_historico.index, df_historico["arrecadacao"], color=CORES["Real"], linewidth=1.6, label="Real")

    for nome in previsoes_teste:
        y_teste = previsoes_teste[nome]
        y_futuro = previsoes_futuras[nome]
        datas_modelo = list(datas_teste) + list(datas_futuras)
        valores_modelo = list(y_teste) + list(y_futuro)
        ax.plot(datas_modelo, valores_modelo, linestyle="--", linewidth=1.2, color=CORES.get(nome), label=nome)

    ax.axvline(datas_teste[0], color="gray", linestyle=":", linewidth=1)
    ax.text(datas_teste[0], df_historico["arrecadacao"].max() * 0.97, "  início do teste", fontsize=9, color="gray")
    ax.axvline(df_historico.index[-1], color="gray", linestyle=":", linewidth=1)
    ax.text(df_historico.index[-1], df_historico["arrecadacao"].max() * 0.90, "  início da previsão futura", fontsize=9, color="gray")

    ano_ini, ano_fim = df_historico.index[0].year, datas_futuras[-1].year
    ax.set_title(
        f"Série histórica completa com previsões de todas as arquiteturas sobrepostas ({ano_ini}–{ano_fim})\n"
        "(período de teste: walk-forward um passo à frente; após o fim da série: previsão multi-step)"
    )
    _fmt_reais(ax)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
