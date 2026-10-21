"""
custo_computacional.py

Mede o custo computacional das três arquiteturas comparadas no Capítulo 3 do
Projeto de Qualificação: tempo de treinamento, tempo de inferência (previsão
multi-step de 12 meses) e complexidade do modelo (nº de parâmetros/árvores).

Não decide qual arquitetura é "melhor": apenas registra, de forma
reproduzível, o custo associado a cada uma, para subsidiar a discussão de
viabilidade prática (Seção 1.5 e Capítulo 5) com números reais, e não apenas
com a acurácia (MAE/RMSE/MAPE) já reportada em outputs/results/metricas_teste.csv.

Uso:
    python scripts/custo_computacional.py
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from src.data.real_data_loader import carregar_serie_real
from src.data.preprocessing import (
    train_test_split_temporal,
    construir_features_lag,
    construir_janelas,
    normalizar,
)
from src.models.classical import walk_forward_sarimax, forecast_multistep_sarimax
from src.models.xgboost_model import treinar_xgboost, forecast_multistep_xgboost
from src.models.lstm_numpy import LSTMForecaster

MESES_TESTE = 12
JANELA_LSTM = 12
EPOCAS_LSTM = 150
N_REPETICOES = 3  # repete cada medição para reduzir ruído de tempo


def medir(func, *args, repeticoes=N_REPETICOES, **kwargs):
    tempos = []
    resultado = None
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        resultado = func(*args, **kwargs)
        tempos.append(time.perf_counter() - t0)
    return resultado, float(np.mean(tempos)), float(np.std(tempos))


def main():
    df = carregar_serie_real()
    treino, teste = train_test_split_temporal(df, meses_teste=MESES_TESTE)
    linhas = []

    # --- SARIMA/SARIMAX ----------------------------------------------------
    _, t_treino_media, t_treino_dp = medir(
        walk_forward_sarimax, treino["arrecadacao"], teste["arrecadacao"], repeticoes=N_REPETICOES
    )
    _, t_infer_media, t_infer_dp = medir(
        forecast_multistep_sarimax, df["arrecadacao"], n_passos=12, repeticoes=N_REPETICOES
    )
    linhas.append({
        "Modelo": "SARIMA/SARIMAX",
        "Nº de parâmetros/árvores": "5 (ordem (1,1,1)x(1,1,1,12) + variância residual)",
        "Tempo de treino+walk-forward (s)": t_treino_media,
        "Desvio-padrão treino (s)": t_treino_dp,
        "Tempo de previsão multi-step 12m (s)": t_infer_media,
    })

    # --- LSTM (NumPy) --------------------------------------------------------
    serie_treino_norm, media, desvio = normalizar(treino["arrecadacao"].values)
    X_treino, y_treino = construir_janelas(serie_treino_norm, janela=JANELA_LSTM)

    def treinar_lstm():
        lstm = LSTMForecaster(hidden_size=16, lr=0.02, seed=42)
        lstm.treinar(X_treino, y_treino, epocas=EPOCAS_LSTM, verbose=False)
        return lstm

    lstm_treinada, t_treino_media, t_treino_dp = medir(treinar_lstm, repeticoes=N_REPETICOES)

    n_parametros = sum(getattr(lstm_treinada, p).size for p in lstm_treinada._params)

    def prever_multistep_lstm():
        serie_completa_norm, media2, desvio2 = normalizar(df["arrecadacao"].values)
        janela_atual = serie_completa_norm[-JANELA_LSTM:].tolist()
        for _ in range(12):
            p = lstm_treinada.prever(np.array(janela_atual[-JANELA_LSTM:]))
            janela_atual.append(p)

    _, t_infer_media, t_infer_dp = medir(prever_multistep_lstm, repeticoes=N_REPETICOES)

    linhas.append({
        "Modelo": "LSTM (NumPy)",
        "Nº de parâmetros/árvores": f"{n_parametros} parâmetros ({EPOCAS_LSTM} épocas, hidden=16)",
        "Tempo de treino+walk-forward (s)": t_treino_media,
        "Desvio-padrão treino (s)": t_treino_dp,
        "Tempo de previsão multi-step 12m (s)": t_infer_media,
    })

    # --- XGBoost -------------------------------------------------------------
    df_feats = construir_features_lag(df)
    treino_feats, _ = train_test_split_temporal(df_feats, meses_teste=MESES_TESTE)

    modelo_xgb, t_treino_media, t_treino_dp = medir(treinar_xgboost, treino_feats, repeticoes=N_REPETICOES)
    n_arvores = modelo_xgb.get_booster().num_boosted_rounds()

    _, t_infer_media, t_infer_dp = medir(
        forecast_multistep_xgboost, modelo_xgb, df_feats, n_passos=12, repeticoes=N_REPETICOES
    )

    linhas.append({
        "Modelo": "XGBoost",
        "Nº de parâmetros/árvores": f"{n_arvores} árvores (max_depth=3)",
        "Tempo de treino+walk-forward (s)": t_treino_media,
        "Desvio-padrão treino (s)": t_treino_dp,
        "Tempo de previsão multi-step 12m (s)": t_infer_media,
    })

    df_custo = pd.DataFrame(linhas)
    saida = RAIZ / "outputs" / "results" / "custo_computacional.csv"
    df_custo.to_csv(saida, index=False)

    print("Ambiente de execução:")
    print(f"  CPU: {platform.processor()}")
    print(f"  SO : {platform.system()} {platform.release()}")
    print(f"  Sem uso de GPU (todas as três arquiteturas rodam em CPU)")
    print()
    print(df_custo.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSalvo em {saida}")


if __name__ == "__main__":
    main()
