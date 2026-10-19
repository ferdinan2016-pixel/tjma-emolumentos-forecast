"""
xgboost_model.py

Modelo 3: XGBoost sobre features de defasagem (lags), conforme Seção 3.3.3
do Projeto de Qualificação - escolhido, no cenário de dados escassos, por
sua robustez reconhecida na literatura frente a redes profundas.

As entradas derivam apenas da própria série de arrecadação (defasagens e
média móvel) e do calendário (sazonalidade cíclica do mês), de modo que as
três arquiteturas comparadas usem a mesma informação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FEATURES = ["lag_1", "lag_2", "lag_3", "lag_12", "media_movel_3", "mes_sin", "mes_cos"]


def treinar_xgboost(df_treino_feats: pd.DataFrame, coluna_alvo: str = "arrecadacao") -> XGBRegressor:
    dados = df_treino_feats.dropna(subset=FEATURES + [coluna_alvo])
    modelo = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    modelo.fit(dados[FEATURES], dados[coluna_alvo])
    return modelo


def walk_forward_xgboost(modelo: XGBRegressor, df_teste_feats: pd.DataFrame) -> np.ndarray:
    """Previsão um passo à frente para o conjunto de teste: como as features
    de defasagem já usam apenas valores reais anteriores a cada mês (walk-
    forward com histórico real disponível), a previsão é direta, sem
    recursão."""
    return modelo.predict(df_teste_feats[FEATURES])


def forecast_multistep_xgboost(
    modelo: XGBRegressor,
    df_completo_feats: pd.DataFrame,
    n_passos: int,
    coluna_alvo: str = "arrecadacao",
) -> np.ndarray:
    """Previsão recursiva multi-step: a cada mês futuro, usa as previsões
    anteriores do próprio modelo como novas defasagens (estratégia recursiva
    padrão em previsão multi-step com modelos de regressão)."""
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
