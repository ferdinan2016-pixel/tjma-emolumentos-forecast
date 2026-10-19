"""
real_data_loader.py

Carrega a série histórica REAL de arrecadação bruta mensal das serventias
extrajudiciais do TJMA, consolidada a partir das publicações mensais oficiais
"Arrecadação Mensal das Serventias Extrajudiciais" (FERJ/TJMA) e do arquivo
histórico consolidado "faturamento_serventias_2019_2024_corrigido".

Cobertura: janeiro/2019 a setembro/2026 (93 meses), sem lacunas.

Metodologia de consolidação (ver data/raw/_parse_arrecadacao_pdfs.py e
_consolidar_serie.py para o código de extração original):
    - Cada publicação mensal lista, linha a linha, a Arrecadação Bruta (A) de
      cada uma das ~280-290 serventias extrajudiciais do Estado.
    - Somou-se a Arrecadação Bruta (A) de todas as serventias em cada mês,
      produzindo a série agregada em nível estadual (cenário-base definido na
      Seção 3.1 do Projeto de Qualificação).
    - Dois meses do período 2017-2018 (dezembro/2017 e dezembro/2018)
      apresentaram inconsistência de formatação na extração (mistura de dois
      padrões de tabela no mesmo PDF) e foram excluídos desta versão; о
      período 2019-2026 não apresentou nenhuma inconsistência dessa natureza.

Esta função substitui `gerar_serie_sintetica()` no pipeline (ver
scripts/run_pipeline.py), conforme planejado na Seção "Como migrar para
dados reais" do README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CAMINHO_PADRAO = Path(__file__).resolve().parents[2] / "data" / "raw" / "arrecadacao_bruta_mensal_real_2019_2026.csv"

# Anos de quebra normativa relevante identificados no Capítulo 2 do Projeto
# de Qualificação (Lei nº 11.400/2020 e Lei nº 12.325/2024). A coluna
# `dummy_lei` fica disponível para experimentos, mas não é usada por nenhum
# dos três modelos da comparação principal.
QUEBRAS_LEGISLATIVAS = {2020, 2024}


def carregar_serie_real(caminho_csv: str | Path = CAMINHO_PADRAO) -> pd.DataFrame:
    """Carrega a série real e devolve um DataFrame no mesmo formato produzido
    por `gerar_serie_sintetica()`, para uso direto no restante da pipeline.

    Colunas devolvidas:
        arrecadacao   -> Arrecadação Bruta (A) somada de todas as serventias no mês (R$)
        dummy_lei     -> 1 nos anos de quebra legislativa adicional, 0 caso contrário
        mes           -> mês do ano (1-12)
        n_serventias  -> nº de serventias que reportaram arrecadação naquele mês (contexto)
    """
    df = pd.read_csv(caminho_csv, parse_dates=["competencia"])
    df = df.set_index("competencia").sort_index()
    df.index.name = "competencia"

    df["dummy_lei"] = [1 if data.year in QUEBRAS_LEGISLATIVAS else 0 for data in df.index]
    df = df.rename(columns={"arrecadacao_bruta": "arrecadacao"})
    df = df[["arrecadacao", "dummy_lei", "mes", "n_serventias"]]
    return df


if __name__ == "__main__":
    df = carregar_serie_real()
    print(df.head(15))
    print("...")
    print(df.tail(5))
    print(f"\nTotal de observações: {len(df)}  ({df.index[0].date()} a {df.index[-1].date()})")
