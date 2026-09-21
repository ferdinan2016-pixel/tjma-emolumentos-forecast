"""
run_pipeline.py

Ponto de entrada único da pipeline experimental descrita no Capítulo 3 do
Projeto de Qualificação:

    1. Geração/carga dos dados (atualmente: série SINTÉTICA - placeholder)
    2. Pré-processamento (features de defasagem, janelas, split temporal)
    3. Treinamento dos três modelos (SARIMA, LSTM, XGBoost)
    4. Avaliação walk-forward (T+1) no conjunto de teste (2025)
    5. Previsão multi-step do ano seguinte (2026)
    6. Geração de gráficos e tabela de métricas em outputs/

Uso:
    python scripts/run_pipeline.py

Não recebe argumentos: todos os parâmetros relevantes estão documentados
nos módulos de src/. Ao substituir a série sintética pela série real do
TJMA, basta trocar a chamada `gerar_serie_sintetica()` por uma função de
carga da série real (ex.: `carregar_serie_ferj(caminho_csv)`), mantendo o
restante da pipeline inalterado.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from src.data.synthetic_generator import gerar_serie_sintetica
from src.data.preprocessing import (
    train_test_split_temporal,
    construir_features_lag,
    construir_janelas,
    normalizar,
    desnormalizar,
)
from src.evaluation.metrics import resumo_metricas
from src.models.classical import walk_forward_sarimax, forecast_multistep_sarimax
from src.models.xgboost_model import treinar_xgboost, walk_forward_xgboost, forecast_multistep_xgboost
from src.models.lstm_numpy import LSTMForecaster
from src.visualization.plots import (
    plot_serie_geral,
    plot_comparacao_teste,
    plot_metricas,
    plot_forecast_proximo_ano,
)

MESES_TESTE = 12
JANELA_LSTM = 12
EPOCAS_LSTM = 150


def main():
    inicio = time.time()
    print("=" * 78)
    print("PIPELINE DE PREVISÃO DA ARRECADAÇÃO DE EMOLUMENTOS EXTRAJUDICIAIS — TJMA")
    print("(prova de conceito com série SINTÉTICA — ver README.md)")
    print("=" * 78)

    # 1) Dados -----------------------------------------------------------
    print("\n[1/6] Gerando série sintética...")
    df = gerar_serie_sintetica()
    df.to_csv(RAIZ / "data" / "processed" / "serie_sintetica.csv")
    print(f"      {len(df)} observações mensais ({df.index[0].date()} a {df.index[-1].date()})")

    treino, teste = train_test_split_temporal(df, meses_teste=MESES_TESTE)
    print(f"      treino: {len(treino)} meses | teste: {len(teste)} meses (ano de 2025)")

    resultados_teste = {}

    # 2) Modelo 1 — SARIMAX (baseline clássico) --------------------------
    print("\n[2/6] Treinando modelo clássico (SARIMAX)...")
    t0 = time.time()
    pred_sarima = walk_forward_sarimax(treino["arrecadacao"], teste["arrecadacao"])
    resultados_teste["ARIMA/SARIMA"] = pred_sarima
    print(f"      concluído em {time.time()-t0:.1f}s")

    # 3) Modelo 2 — LSTM (NumPy) ------------------------------------------
    print("\n[3/6] Treinando LSTM (implementação NumPy)...")
    t0 = time.time()
    serie_treino_norm, media, desvio = normalizar(treino["arrecadacao"].values)
    X_treino, y_treino = construir_janelas(serie_treino_norm, janela=JANELA_LSTM)

    lstm = LSTMForecaster(hidden_size=16, lr=0.02, seed=42)
    historico_loss = lstm.treinar(X_treino, y_treino, epocas=EPOCAS_LSTM, verbose=True)

    serie_completa_norm = (df["arrecadacao"].values - media) / desvio
    idx_inicio_teste = len(treino) - JANELA_LSTM
    pred_lstm_norm = []
    for i in range(MESES_TESTE):
        janela = serie_completa_norm[idx_inicio_teste + i : idx_inicio_teste + i + JANELA_LSTM]
        pred_lstm_norm.append(lstm.prever(janela))
    pred_lstm = desnormalizar(np.array(pred_lstm_norm), media, desvio)
    resultados_teste["LSTM"] = pred_lstm
    print(f"      concluído em {time.time()-t0:.1f}s ({EPOCAS_LSTM} épocas)")

    # 4) Modelo 3 — XGBoost -----------------------------------------------
    print("\n[4/6] Treinando XGBoost (features de defasagem)...")
    t0 = time.time()
    df_feats = construir_features_lag(df)
    treino_feats, teste_feats = train_test_split_temporal(df_feats, meses_teste=MESES_TESTE)
    modelo_xgb = treinar_xgboost(treino_feats)
    pred_xgb = walk_forward_xgboost(modelo_xgb, teste_feats)
    resultados_teste["XGBoost"] = pred_xgb
    print(f"      concluído em {time.time()-t0:.1f}s")

    # 5) Avaliação ----------------------------------------------------------
    print("\n[5/6] Avaliando modelos no conjunto de teste (2025)...")
    y_real = teste["arrecadacao"].values
    linhas_metricas = {nome: resumo_metricas(y_real, pred) for nome, pred in resultados_teste.items()}
    df_metricas = pd.DataFrame(linhas_metricas).T
    df_metricas = df_metricas[["MAE", "RMSE", "MAPE (%)"]].sort_values("MAPE (%)")
    print(df_metricas.to_string(float_format=lambda v: f"{v:,.2f}"))
    df_metricas.to_csv(RAIZ / "outputs" / "results" / "metricas_teste_2025.csv")

    df_predicoes = pd.DataFrame({"real": y_real}, index=teste.index)
    for nome, pred in resultados_teste.items():
        df_predicoes[nome] = pred
    df_predicoes.to_csv(RAIZ / "outputs" / "results" / "predicoes_teste_2025.csv")

    # 6) Previsão do "ano seguinte" (2026) — multi-step ---------------------
    print("\n[6/6] Gerando previsão multi-step para 2026 (ano seguinte)...")
    datas_futuras = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")

    pred_sarima_2026 = forecast_multistep_sarimax(df["arrecadacao"], n_passos=12)

    serie_completa_norm2, media2, desvio2 = normalizar(df["arrecadacao"].values)
    janela_atual = serie_completa_norm2[-JANELA_LSTM:].tolist()
    pred_lstm_2026_norm = []
    for _ in range(12):
        p = lstm.prever(np.array(janela_atual[-JANELA_LSTM:]))
        pred_lstm_2026_norm.append(p)
        janela_atual.append(p)
    pred_lstm_2026 = desnormalizar(np.array(pred_lstm_2026_norm), media2, desvio2)

    pred_xgb_2026 = forecast_multistep_xgboost(modelo_xgb, df_feats, n_passos=12)

    df_forecast_2026 = pd.DataFrame(
        {
            "ARIMA/SARIMA": pred_sarima_2026,
            "LSTM": pred_lstm_2026,
            "XGBoost": pred_xgb_2026,
        },
        index=datas_futuras,
    )
    df_forecast_2026.to_csv(RAIZ / "outputs" / "results" / "previsao_2026.csv")
    print(df_forecast_2026.apply(lambda c: c.map(lambda v: f"R$ {v:,.0f}")).to_string())

    total_2026 = df_forecast_2026.sum()
    print("\n      Total anual previsto para 2026, por modelo:")
    for nome, valor in total_2026.items():
        print(f"        {nome:<14s}: R$ {valor:,.2f}")

    # Gráficos ---------------------------------------------------------------
    print("\nGerando gráficos em outputs/figures/ ...")
    fig_dir = RAIZ / "outputs" / "figures"
    plot_serie_geral(df, teste.index[0], str(fig_dir / "01_serie_geral.png"))
    plot_comparacao_teste(teste.index, y_real, resultados_teste, str(fig_dir / "02_comparacao_teste_2025.png"))
    plot_metricas(df_metricas, str(fig_dir / "03_comparacao_metricas.png"))
    plot_forecast_proximo_ano(df, datas_futuras, {
        "ARIMA/SARIMA": pred_sarima_2026, "LSTM": pred_lstm_2026, "XGBoost": pred_xgb_2026,
    }, str(fig_dir / "04_previsao_2026.png"))

    print(f"\nPipeline concluída em {time.time()-inicio:.1f}s.")
    print("Resultados em outputs/results/  |  Gráficos em outputs/figures/")


if __name__ == "__main__":
    main()
