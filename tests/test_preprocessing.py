import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (
    train_test_split_temporal,
    construir_janelas,
    normalizar,
    desnormalizar,
)


def test_split_temporal_preserva_ordem_e_tamanho():
    df = pd.DataFrame({"x": range(24)}, index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    treino, teste = train_test_split_temporal(df, meses_teste=12)
    assert len(treino) == 12
    assert len(teste) == 12
    assert treino.index[-1] < teste.index[0]


def test_construir_janelas_formato():
    serie = np.arange(20)
    X, y = construir_janelas(serie, janela=5)
    assert X.shape == (15, 5)
    assert y.shape == (15,)
    assert list(X[0]) == [0, 1, 2, 3, 4]
    assert y[0] == 5


def test_normalizar_desnormalizar_e_inversa():
    serie = np.array([10.0, 20.0, 30.0, 40.0])
    norm, media, desvio = normalizar(serie)
    reconstruida = desnormalizar(norm, media, desvio)
    assert np.allclose(serie, reconstruida)


def test_normalizar_usa_parametros_do_treino_no_teste():
    treino = np.array([10.0, 20.0, 30.0])
    teste = np.array([40.0])
    _, media, desvio = normalizar(treino)
    teste_norm, media2, desvio2 = normalizar(teste, media=media, desvio=desvio)
    assert media2 == media and desvio2 == desvio


if __name__ == "__main__":
    test_split_temporal_preserva_ordem_e_tamanho()
    test_construir_janelas_formato()
    test_normalizar_desnormalizar_e_inversa()
    test_normalizar_usa_parametros_do_treino_no_teste()
    print("OK - todos os testes de preprocessing.py passaram")
