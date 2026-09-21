"""
classical.py

Modelo 1 (baseline): SARIMA/SARIMAX (statsmodels), conforme Seção 3.3.1 do
Projeto de Qualificação. Realiza previsão walk-forward um passo à frente
(T+1), atualizando o modelo com a observação real a cada mês, prática usual
em previsão de receita pública (ex.: modelo LSTM T+1 da SEFAZ-RS citado no
Capítulo 2), e também uma previsão multi-step (12 meses) diretamente a
partir do fim da série histórica, para o caso de uso "prever a arrecadação
do ano seguinte".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def walk_forward_sarimax(
    treino: pd.Series,
    teste: pd.Series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
) -> np.ndarray:
    """Ajusta o SARIMAX no treino e produz previsões um passo à frente (T+1)
    para cada mês do conjunto de teste, incorporando a cada iteração a
    observação real do mês anterior (simulação realista de uso em produção).
    """
    modelo = SARIMAX(
        treino, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    resultado = modelo.fit(disp=False)

    previsoes = []
    historico = treino.copy()
    for data, valor_real in teste.items():
        pred = resultado.get_forecast(steps=1).predicted_mean.iloc[0]
        previsoes.append(pred)
        historico = pd.concat([historico, pd.Series([valor_real], index=[data])])
        resultado = resultado.append([valor_real], refit=False)

    return np.array(previsoes)


def forecast_multistep_sarimax(
    serie_completa: pd.Series,
    n_passos: int = 12,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
) -> np.ndarray:
    """Previsão multi-step direta (sem observar o real a cada mês) para os
    próximos `n_passos` meses a partir do fim da série completa - representa
    o caso de uso real de "prever a arrecadação do ano seguinte" descrito no
    objetivo geral do projeto."""
    modelo = SARIMAX(
        serie_completa, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    resultado = modelo.fit(disp=False)
    previsao = resultado.get_forecast(steps=n_passos).predicted_mean
    return previsao.values
