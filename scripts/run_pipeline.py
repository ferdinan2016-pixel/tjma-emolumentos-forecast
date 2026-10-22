"""
run_pipeline.py

Ponto de entrada único da pipeline experimental descrita no Capítulo 3 do
Projeto de Qualificação:

    1. Carga dos dados (padrão: série REAL de arrecadação do FERJ/TJMA,
       2019-2026; opção: série sintética, mantida para referência)
    2. Pré-processamento (features de defasagem, janelas, split temporal)
    3. Treinamento dos sete modelos (SARIMA, LSTM, GRU, XGBoost, Random
       Forest, LightGBM, CatBoost)
    4. Avaliação walk-forward (T+1) nos últimos 12 meses observados
    5. Previsão multi-step dos 12 meses seguintes ("ano seguinte")
    6. Geração de gráficos e tabela de métricas em outputs/

Uso:
    python scripts/run_pipeline.py            # usa a série REAL (padrão)
    python scripts/run_pipeline.py sintetica   # usa a série sintética (placeholder)
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
from src.data.real_data_loader import carregar_serie_real
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
from src.models.random_forest_model import (
    treinar_random_forest, walk_forward_random_forest, forecast_multistep_random_forest,
)
from src.models.lightgbm_model import (
    treinar_lightgbm, walk_forward_lightgbm, forecast_multistep_lightgbm,
)
from src.models.catboost_model import (
    treinar_catboost, walk_forward_catboost, forecast_multistep_catboost,
)
from src.models.lstm_numpy import LSTMForecaster
from src.models.gru_numpy import GRUForecaster
from src.visualization.plots import (
    plot_serie_geral,
    plot_comparacao_teste,
    plot_metricas,
    plot_forecast_proximo_ano,
    plot_serie_completa_com_previsoes,
)

MESES_TESTE = 12
JANELA_LSTM = 12
EPOCAS_LSTM = 150


def main(fonte: str = "real"):
    """fonte: "real" (padrão, série FERJ/TJMA 2019-2026) ou "sintetica"
    (placeholder anterior, mantido para referência/comparação)."""
    inicio = time.time()
    print("=" * 78)
    print("PIPELINE DE PREVISÃO DA ARRECADAÇÃO DE EMOLUMENTOS EXTRAJUDICIAIS — TJMA")
    if fonte == "real":
        print("(dados REAIS: arrecadação bruta mensal consolidada do FERJ/TJMA, 2019-2026)")
    else:
        print("(prova de conceito com série SINTÉTICA — ver README.md)")
    print("=" * 78)

    sufixo = "" if fonte == "real" else "_sintetico"

    # 1) Dados -----------------------------------------------------------
    print(f"\n[1/6] Carregando série ({fonte})...")
    df = carregar_serie_real() if fonte == "real" else gerar_serie_sintetica()
    nome_arquivo = "serie_real.csv" if fonte == "real" else "serie_sintetica.csv"
    df.to_csv(RAIZ / "data" / "processed" / nome_arquivo)
    print(f"      {len(df)} observações mensais ({df.index[0].date()} a {df.index[-1].date()})")

    treino, teste = train_test_split_temporal(df, meses_teste=MESES_TESTE)
    print(f"      treino: {len(treino)} meses ({treino.index[0].date()} a {treino.index[-1].date()})")
    print(f"      teste : {len(teste)} meses ({teste.index[0].date()} a {teste.index[-1].date()})")

    resultados_teste = {}

    # 2) Modelo 1 — SARIMAX (baseline clássico) --------------------------
    print("\n[2/8] Treinando modelo clássico (SARIMAX)...")
    t0 = time.time()
    pred_sarima = walk_forward_sarimax(treino["arrecadacao"], teste["arrecadacao"])
    resultados_teste["ARIMA/SARIMA"] = pred_sarima
    print(f"      concluído em {time.time()-t0:.1f}s")

    # 3) Modelo 2 — LSTM (NumPy) ------------------------------------------
    print("\n[3/8] Treinando LSTM (implementação NumPy)...")
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

    # 4) Modelo 3 — GRU (NumPy) -------------------------------------------
    print("\n[4/8] Treinando GRU (implementação NumPy)...")
    t0 = time.time()
    gru = GRUForecaster(hidden_size=16, lr=0.02, seed=42)
    gru.treinar(X_treino, y_treino, epocas=EPOCAS_LSTM, verbose=True)

    pred_gru_norm = []
    for i in range(MESES_TESTE):
        janela = serie_completa_norm[idx_inicio_teste + i : idx_inicio_teste + i + JANELA_LSTM]
        pred_gru_norm.append(gru.prever(janela))
    pred_gru = desnormalizar(np.array(pred_gru_norm), media, desvio)
    resultados_teste["GRU"] = pred_gru
    print(f"      concluído em {time.time()-t0:.1f}s ({EPOCAS_LSTM} épocas)")

    # 5) Modelos 4-7 — árvores sobre features de defasagem ------------------
    df_feats = construir_features_lag(df)
    treino_feats, teste_feats = train_test_split_temporal(df_feats, meses_teste=MESES_TESTE)

    print("\n[5/8] Treinando XGBoost (features de defasagem)...")
    t0 = time.time()
    modelo_xgb = treinar_xgboost(treino_feats)
    pred_xgb = walk_forward_xgboost(modelo_xgb, teste_feats)
    resultados_teste["XGBoost"] = pred_xgb
    print(f"      concluído em {time.time()-t0:.1f}s")

    print("\n[6/8] Treinando Random Forest (features de defasagem)...")
    t0 = time.time()
    modelo_rf = treinar_random_forest(treino_feats)
    pred_rf = walk_forward_random_forest(modelo_rf, teste_feats)
    resultados_teste["Random Forest"] = pred_rf
    print(f"      concluído em {time.time()-t0:.1f}s")

    print("\n[7/8] Treinando LightGBM (features de defasagem)...")
    t0 = time.time()
    modelo_lgbm = treinar_lightgbm(treino_feats)
    pred_lgbm = walk_forward_lightgbm(modelo_lgbm, teste_feats)
    resultados_teste["LightGBM"] = pred_lgbm
    print(f"      concluído em {time.time()-t0:.1f}s")

    print("\n[8/8] Treinando CatBoost (features de defasagem)...")
    t0 = time.time()
    modelo_cat = treinar_catboost(treino_feats)
    pred_cat = walk_forward_catboost(modelo_cat, teste_feats)
    resultados_teste["CatBoost"] = pred_cat
    print(f"      concluído em {time.time()-t0:.1f}s")

    # 6) Avaliação ----------------------------------------------------------
    print(f"\nAvaliando modelos no conjunto de teste ({teste.index[0].date()} a {teste.index[-1].date()})...")
    y_real = teste["arrecadacao"].values
    linhas_metricas = {nome: resumo_metricas(y_real, pred) for nome, pred in resultados_teste.items()}
    df_metricas = pd.DataFrame(linhas_metricas).T
    df_metricas = df_metricas[["MAE", "RMSE", "MAPE (%)"]].sort_values("MAPE (%)")
    print(df_metricas.to_string(float_format=lambda v: f"{v:,.2f}"))
    df_metricas.to_csv(RAIZ / "outputs" / "results" / f"metricas_teste{sufixo}.csv")

    df_predicoes = pd.DataFrame({"real": y_real}, index=teste.index)
    for nome, pred in resultados_teste.items():
        df_predicoes[nome] = pred
    df_predicoes.to_csv(RAIZ / "outputs" / "results" / f"predicoes_teste{sufixo}.csv")

    # 7) Previsão dos 12 meses seguintes — multi-step ------------------------
    datas_futuras = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
    print(f"\nGerando previsão multi-step para {datas_futuras[0].date()} a {datas_futuras[-1].date()}...")

    pred_sarima_fut = forecast_multistep_sarimax(df["arrecadacao"], n_passos=12)

    serie_completa_norm2, media2, desvio2 = normalizar(df["arrecadacao"].values)

    janela_lstm_fut = serie_completa_norm2[-JANELA_LSTM:].tolist()
    pred_lstm_fut_norm = []
    for _ in range(12):
        p = lstm.prever(np.array(janela_lstm_fut[-JANELA_LSTM:]))
        pred_lstm_fut_norm.append(p)
        janela_lstm_fut.append(p)
    pred_lstm_fut = desnormalizar(np.array(pred_lstm_fut_norm), media2, desvio2)

    janela_gru_fut = serie_completa_norm2[-JANELA_LSTM:].tolist()
    pred_gru_fut_norm = []
    for _ in range(12):
        p = gru.prever(np.array(janela_gru_fut[-JANELA_LSTM:]))
        pred_gru_fut_norm.append(p)
        janela_gru_fut.append(p)
    pred_gru_fut = desnormalizar(np.array(pred_gru_fut_norm), media2, desvio2)

    pred_xgb_fut = forecast_multistep_xgboost(modelo_xgb, df_feats, n_passos=12)
    pred_rf_fut = forecast_multistep_random_forest(modelo_rf, df_feats, n_passos=12)
    pred_lgbm_fut = forecast_multistep_lightgbm(modelo_lgbm, df_feats, n_passos=12)
    pred_cat_fut = forecast_multistep_catboost(modelo_cat, df_feats, n_passos=12)

    previsoes_futuras = {
        "ARIMA/SARIMA": pred_sarima_fut,
        "LSTM": pred_lstm_fut,
        "GRU": pred_gru_fut,
        "XGBoost": pred_xgb_fut,
        "Random Forest": pred_rf_fut,
        "LightGBM": pred_lgbm_fut,
        "CatBoost": pred_cat_fut,
    }

    df_forecast_fut = pd.DataFrame(previsoes_futuras, index=datas_futuras)
    df_forecast_fut.to_csv(RAIZ / "outputs" / "results" / f"previsao_proximos_12_meses{sufixo}.csv")
    print(df_forecast_fut.apply(lambda c: c.map(lambda v: f"R$ {v:,.0f}")).to_string())

    total_fut = df_forecast_fut.sum()
    print("\n      Total previsto para os próximos 12 meses, por modelo:")
    for nome, valor in total_fut.items():
        print(f"        {nome:<14s}: R$ {valor:,.2f}")

    # Gráficos ---------------------------------------------------------------
    print("\nGerando gráficos em outputs/figures/ ...")
    fig_dir = RAIZ / "outputs" / "figures"
    plot_serie_geral(df, teste.index[0], str(fig_dir / f"01_serie_geral{sufixo}.png"), fonte=fonte)
    plot_comparacao_teste(teste.index, y_real, resultados_teste, str(fig_dir / f"02_comparacao_teste{sufixo}.png"))
    plot_metricas(df_metricas, str(fig_dir / f"03_comparacao_metricas{sufixo}.png"))
    plot_forecast_proximo_ano(df, datas_futuras, previsoes_futuras, str(fig_dir / f"04_previsao_futura{sufixo}.png"))
    plot_serie_completa_com_previsoes(
        df, teste.index, resultados_teste, datas_futuras, previsoes_futuras,
        str(fig_dir / f"05_serie_completa_previsoes{sufixo}.png"),
    )

    print(f"\nPipeline concluída em {time.time()-inicio:.1f}s.")
    print("Resultados em outputs/results/  |  Gráficos em outputs/figures/")


if __name__ == "__main__":
    fonte_cli = sys.argv[1] if len(sys.argv) > 1 else "real"
    main(fonte=fonte_cli)
