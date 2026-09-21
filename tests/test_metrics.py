import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import mae, rmse, mape, resumo_metricas


def test_mae_zero_quando_previsao_perfeita():
    y = np.array([100.0, 200.0, 300.0])
    assert mae(y, y) == 0.0


def test_rmse_zero_quando_previsao_perfeita():
    y = np.array([100.0, 200.0, 300.0])
    assert rmse(y, y) == 0.0


def test_mape_calculo_simples():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    assert abs(mape(y_true, y_pred) - 10.0) < 1e-9


def test_resumo_metricas_contem_as_tres_chaves():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 210.0])
    resumo = resumo_metricas(y_true, y_pred)
    assert set(resumo.keys()) == {"MAE", "RMSE", "MAPE (%)"}
    assert all(v >= 0 for v in resumo.values())


if __name__ == "__main__":
    test_mae_zero_quando_previsao_perfeita()
    test_rmse_zero_quando_previsao_perfeita()
    test_mape_calculo_simples()
    test_resumo_metricas_contem_as_tres_chaves()
    print("OK - todos os testes de metrics.py passaram")
