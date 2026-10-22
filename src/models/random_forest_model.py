"""
random_forest_model.py

Modelo adicional: Random Forest sobre as mesmas features de defasagem (lags)
usadas pelo XGBoost (Seção 3.3 do Projeto de Qualificação), incluído na
comparação a pedido do orientador como representante de ensembles de árvores
por bagging (em contraste com o XGBoost, LightGBM e CatBoost, que são
ensembles por boosting).

Segue exatamente a mesma interface (treinar / walk_forward / forecast
multi-step) e o mesmo conjunto de features de xgboost_model.py, para que a
comparação entre arquiteturas de árvore seja direta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["lag_1", "lag_2", "lag_3", "lag_12", "media_movel_3", "mes_sin", "mes_cos"]


def treinar_random_forest(df_treino_feats: pd.DataFrame, coluna_alvo: str = "arrecadacao") -> RandomForestRegressor:
    dados = df_treino_feats.dropna(subset=FEATURES + [coluna_alvo])
    modelo = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(dados[FEATURES], dados[coluna_alvo])
    return modelo


def walk_forward_random_forest(modelo: RandomForestRegressor, df_teste_feats: pd.DataFrame) -> np.ndarray:
    return modelo.predict(df_teste_feats[FEATURES])


def forecast_multistep_random_forest(
    modelo: RandomForestRegressor,
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
