"""
analise_estatistica.py

Análise complementar sobre as previsões do conjunto de teste geradas por
run_pipeline.py (outputs/results/predicoes_teste.csv):

    1. Ensembles por média simples de todas as combinações 2 a 2 e 3 a 3 das
       três arquiteturas (procedimento de Silva, Cortes e Diniz, 2023);
    2. Teste de Wilcoxon pareado sobre o erro absoluto mensal e teste de
       Friedman com os meses como blocos (Demšar, 2006);
    3. Diagnóstico de extrapolação: previsões versus o máximo do treino.

Uso:
    python scripts/analise_estatistica.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.evaluation.metrics import resumo_metricas
from src.data.real_data_loader import carregar_serie_real

MODELOS = ["ARIMA/SARIMA", "LSTM", "XGBoost"]


def main():
    pred = pd.read_csv(RAIZ / "outputs" / "results" / "predicoes_teste.csv", index_col=0, parse_dates=True)
    serie = carregar_serie_real()["arrecadacao"]
    y = pred["real"].values

    print("=== 1. Ensembles por media simples (todas as combinacoes) ===")
    linhas = []
    for k in (1, 2, 3):
        for combo in combinations(MODELOS, k):
            m = resumo_metricas(y, pred[list(combo)].mean(axis=1).values)
            linhas.append({"modelo": " + ".join(combo), **m})
    tab = pd.DataFrame(linhas).set_index("modelo").sort_values("MAPE (%)")
    print(tab.to_string(float_format=lambda v: f"{v:,.2f}"))
    tab.to_csv(RAIZ / "outputs" / "results" / "ensembles_teste.csv")

    print("\n=== 2. Wilcoxon pareado sobre o erro absoluto mensal (n=12) ===")
    ea = {m: np.abs(y - pred[m].values) for m in MODELOS}
    ea["Ensemble SARIMA+XGBoost"] = np.abs(y - pred[["ARIMA/SARIMA", "XGBoost"]].mean(axis=1).values)
    pares = list(combinations(MODELOS, 2)) + [
        ("Ensemble SARIMA+XGBoost", "XGBoost"),
        ("Ensemble SARIMA+XGBoost", "ARIMA/SARIMA"),
    ]
    for a, b in pares:
        w, p = wilcoxon(ea[a], ea[b])
        vitorias = int(np.sum(ea[a] < ea[b]))
        print(f"{a} vs {b}: W={w:.1f}  p={p:.3f}  ({a} com menor erro em {vitorias}/12 meses)")

    print("\n=== 3. Friedman (3 arquiteturas, meses como blocos) ===")
    chi2, p = friedmanchisquare(*[ea[m] for m in MODELOS])
    ranks = pd.DataFrame({m: ea[m] for m in MODELOS}).rank(axis=1).mean()
    print(f"chi2={chi2:.3f}  p={p:.3f}")
    print("posto medio (1 = melhor):", {m: round(r, 2) for m, r in ranks.items()})

    print("\n=== 4. Extrapolacao ===")
    treino = serie.loc[: pred.index[0] - pd.DateOffset(months=1)]
    print(f"maximo do treino: R$ {treino.max():,.2f}")
    print(f"maximo real no teste: R$ {y.max():,.2f} ({int(np.sum(y > treino.max()))}/12 meses acima do maximo do treino)")
    for m in MODELOS:
        vies = np.mean(y - pred[m].values)
        print(f"{m}: maior previsao R$ {pred[m].max():,.2f} | vies medio (real - previsto) R$ {vies:,.2f}")


if __name__ == "__main__":
    main()
