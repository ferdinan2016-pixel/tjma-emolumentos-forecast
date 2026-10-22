"""
lightgbm_model.py

Modelo adicional: LightGBM (gradient boosting por crescimento de árvores
leaf-wise) sobre as mesmas features de defasagem (lags) usadas pelo XGBoost
(Seção 3.3 do Projeto de Qualificação), incluído na comparação a pedido do
orientador para ampliar o espectro de arquiteturas de boosting avaliadas.

Segue exatamente a mesma interface (treinar / walk_forward / forecast
multi-step) e o mesmo conjunto de features de xgboost_model.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

FEATURES = ["lag_1", "lag_2", "lag_3", "lag_12", "media_movel_3", "mes_sin", "mes_cos"]


def treinar_lightgbm(df_treino_feats: pd.DataFrame, coluna_alvo: str = "arrecadacao") -> LGBMRegressor:
    dados = df_treino_feats.dropna(subset=FEATURES + [coluna_alvo])
    modelo = LGBMRegressor(
        n_estimators=300,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=5,
        random_state=42,
        verbosity=-1,
    )
    modelo.fit(dados[FEATURES], dados[coluna_alvo])
    return modelo


def walk_forward_lightgbm(modelo: LGBMRegressor, df_teste_feats: pd.DataFrame) -> np.ndarray:
    return modelo.predict(df_teste_feats[FEATURES])


def forecast_multistep_lightgbm(
    modelo: LGBMRegressor,
    df_completo_feats: pd.DataFrame,
    n_passos: int,
    coluna_alvo: str = "arrecadacao",
) -> np.ndarray:
    historico = df_completo_feats[coluna_alvo].copy()
    previsoes = []
    ultimo_mes = df_completo_feats.index[-1].month
    for passo in range(n_passos):
        mes_futuro = (ultimo_mes + passo) % 12 + 1
        linha = {
            "lag_1": historico.iloc[-1],
            "lag_2": historico.iloc[-2],
            "lag_3": historico.iloc[-3],
            "lag_12": historico.iloc[-12],
            "media_movel_3": historico.iloc[-3:].mean(),
            "mes_sin": np.sin(2 * np.pi * mes_futuro / 12),
            "mes_cos": np.cos(2 * np.pi * mes_futuro / 12),
        }
        X = pd.DataFrame([linha])[FEATURES]
        pred = float(modelo.predict(X)[0])
        previsoes.append(pred)
        historico = pd.concat([historico, pd.Series([pred])], ignore_index=True)
    return np.array(previsoes)
