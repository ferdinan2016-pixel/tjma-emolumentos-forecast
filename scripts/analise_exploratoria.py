"""
analise_exploratoria.py

Análise exploratória da série de arrecadação (EDA), solicitada pelo
orientador para demonstrar, com evidência empírica, as escolhas de
modelagem adotadas no Capítulo 3 do Projeto de Qualificação (em particular,
o uso de um modelo sazonal - SARIMA - e de features de defasagem de 12
meses nos modelos de árvore/boosting).

Produz:
    - outputs/figures/00_decomposicao_sazonal.png: decomposição aditiva da
      série (tendência, sazonalidade, resíduo), via STL (Seasonal-Trend
      decomposition using LOESS);
    - outputs/results/analise_exploratoria.csv: estatísticas-resumo (média,
      desvio-padrão, variação ano a ano, teste de estacionariedade ADF,
      força da tendência e da sazonalidade, conforme Hyndman e Athanasopoulos).

Uso:
    python scripts/analise_exploratoria.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from src.data.real_data_loader import carregar_serie_real


def _fmt_reais(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:,.1f} mi"))


def main():
    df = carregar_serie_real()
    serie = df["arrecadacao"]

    # Decomposição STL (período = 12, sazonalidade anual) ---------------------
    stl = STL(serie, period=12, robust=True)
    resultado = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(serie.index, serie.values, color="#1a1a1a", linewidth=1.3)
    axes[0].set_title("Observado")
    axes[1].plot(serie.index, resultado.trend, color="#1f6f4d", linewidth=1.3)
    axes[1].set_title("Tendência")
    axes[2].plot(serie.index, resultado.seasonal, color="#2a4a9e", linewidth=1.1)
    axes[2].set_title("Sazonalidade (período = 12 meses)")
    axes[3].plot(serie.index, resultado.resid, color="#b3401f", linewidth=1.0)
    axes[3].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[3].set_title("Resíduo")
    for ax in axes[:3]:
        _fmt_reais(ax)
    fig.suptitle(
        "Decomposição STL da arrecadação bruta mensal de emolumentos extrajudiciais do TJMA "
        f"({serie.index[0].year}-{serie.index[-1].year})"
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_dir = RAIZ / "outputs" / "figures"
    caminho_fig = fig_dir / "00_decomposicao_sazonal.png"
    fig.savefig(caminho_fig, dpi=150)
    plt.close(fig)
    print(f"Figura salva em {caminho_fig}")

    # Força da tendência e da sazonalidade (Hyndman & Athanasopoulos, 2021) ---
    # F_T = max(0, 1 - Var(resid) / Var(trend + resid))
    # F_S = max(0, 1 - Var(resid) / Var(seasonal + resid))
    var_resid = np.var(resultado.resid)
    forca_tendencia = max(0.0, 1 - var_resid / np.var(resultado.trend + resultado.resid))
    forca_sazonalidade = max(0.0, 1 - var_resid / np.var(resultado.seasonal + resultado.resid))

    # Teste de estacionariedade (Augmented Dickey-Fuller) ----------------------
    adf_stat, adf_pvalue, *_ = adfuller(serie, autolag="AIC", result_object=False)

    # Variação ano a ano (crescimento médio) -----------------------------------
    serie_anual = serie.resample("YS").sum()
    variacao_anual_pct = serie_anual.pct_change().dropna() * 100

    resumo = pd.DataFrame(
        {
            "métrica": [
                "média mensal (R$)",
                "desvio-padrão mensal (R$)",
                "força da tendência (0-1)",
                "força da sazonalidade (0-1)",
                "ADF estatística",
                "ADF p-valor",
                "série estacionária (ADF, alfa=0,05)",
                "variação média ano a ano (%, anos completos)",
            ],
            "valor": [
                f"{serie.mean():,.2f}",
                f"{serie.std():,.2f}",
                f"{forca_tendencia:.3f}",
                f"{forca_sazonalidade:.3f}",
                f"{adf_stat:.3f}",
                f"{adf_pvalue:.4f}",
                "não" if adf_pvalue > 0.05 else "sim",
                f"{variacao_anual_pct.mean():.1f}",
            ],
        }
    )
    caminho_csv = RAIZ / "outputs" / "results" / "analise_exploratoria.csv"
    resumo.to_csv(caminho_csv, index=False)
    print(f"\nResumo estatístico salvo em {caminho_csv}\n")
    print(resumo.to_string(index=False))

    print(
        "\nInterpretação: força da tendência e/ou da sazonalidade próximas de 1 indicam "
        "componente forte e previsível, o que justifica (i) o uso de um modelo sazonal "
        "(SARIMA, Seção 3.3.1) como linha de base, e (ii) a inclusão de defasagem de 12 "
        "meses (lag_12) e de variáveis cíclicas de mês (mes_sin/mes_cos) nos modelos de "
        "árvore/boosting (Seção 3.2). Um p-valor do teste ADF acima de 0,05 indica que a "
        "série não é estacionária em nível, consistente com a tendência de crescimento "
        "observada e com a escolha de diferenciação (d=1) no SARIMA."
    )


if __name__ == "__main__":
    main()
