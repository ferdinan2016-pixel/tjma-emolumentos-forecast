"""
preprocessing.py

Funções de pré-processamento comuns às três arquiteturas: construção de
features de defasagem (lags) para o modelo de árvore/boosting, janelas
deslizantes para a LSTM, e separação treino/teste respeitando a ordem
temporal (nunca embaralhar séries temporais).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_split_temporal(df: pd.DataFrame, meses_teste: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os últimos `meses_teste` meses como conjunto de teste (hold-out),
    preservando a ordem temporal (não faz sentido embaralhar séries temporais).
    """
    treino = df.iloc[:-meses_teste].copy()
    teste = df.iloc[-meses_teste:].copy()
    return treino, teste


def construir_features_lag(
    df: pd.DataFrame,
    coluna_alvo: str = "arrecadacao",
    lags: tuple[int, ...] = (1, 2, 3, 12),
) -> pd.DataFrame:
    """Cria features de defasagem (lags), média móvel de 3 meses e variáveis
    sazonais cíclicas (seno/cosseno do mês), usadas pelo modelo de boosting
    (XGBoost) e como referência para a janela de entrada da LSTM.
    """
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[coluna_alvo].shift(lag)
    out["media_movel_3"] = out[coluna_alvo].shift(1).rolling(window=3).mean()
    out["mes_sin"] = np.sin(2 * np.pi * out["mes"] / 12)
    out["mes_cos"] = np.cos(2 * np.pi * out["mes"] / 12)
    return out


def construir_janelas(
    serie: np.ndarray, janela: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Transforma uma série 1D em pares (X, y) para treino supervisionado de
    uma rede recorrente: X[i] = janela de `janela` valores consecutivos,
    y[i] = valor imediatamente seguinte à janela.
    """
    X, y = [], []
    for i in range(len(serie) - janela):
        X.append(serie[i : i + janela])
        y.append(serie[i + janela])
    return np.array(X), np.array(y)


def normalizar(serie: np.ndarray, media: float | None = None, desvio: float | None = None):
    """Normalização z-score. Se média/desvio não forem fornecidos, são
    calculados a partir da própria série (uso: sempre ajustar apenas no
    conjunto de treino e aplicar os mesmos parâmetros ao teste, evitando
    vazamento de informação futura)."""
    if media is None:
        media = serie.mean()
    if desvio is None:
        desvio = serie.std()
    return (serie - media) / desvio, media, desvio


def desnormalizar(serie_norm, media: float, desvio: float):
    return serie_norm * desvio + media
