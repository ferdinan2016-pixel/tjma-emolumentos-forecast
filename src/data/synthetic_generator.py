"""
synthetic_generator.py

Gera uma série mensal SINTÉTICA de arrecadação de emolumentos extrajudiciais,
usada como placeholder enquanto a série histórica real do TJMA (FERJ/Corregedoria)
não é obtida (ver Capítulo 3 do Projeto de Qualificação).

A série sintética foi desenhada para reproduzir, de forma estilizada, a estrutura
descrita no marco jurídico do projeto:
    - tendência de crescimento real (mais atos praticados ao longo dos anos);
    - sazonalidade mensal (picos e vales recorrentes dentro do ano);
    - reajuste anual da tabela de emolumentos pelo INPC, aplicado em janeiro
      (efeito "preço"), com percentuais plausíveis ano a ano;
    - duas quebras estruturais adicionais, associadas a alterações legislativas
      (Lei nº 11.400/2020 e Lei nº 12.325/2024);
    - ruído aleatório.

Isso permite validar a pipeline completa (ETL -> modelagem -> avaliação -> gráficos)
de ponta a ponta antes da chegada dos dados reais, sem qualquer pretensão de que
os valores numéricos aqui gerados representem a arrecadação real do TJMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Percentuais de reajuste anual (INPC), ano -> % de reajuste em janeiro daquele ano.
# 2025 e 2026 usam os percentuais efetivamente noticiados pelo TJMA (4,84% e 4,17%);
# os demais anos são valores plausíveis simulados dentro da faixa histórica do INPC.
INPC_REAJUSTE = {
    2012: 5.80, 2013: 5.40, 2014: 6.20, 2015: 8.90, 2016: 6.60,
    2017: 2.30, 2018: 3.50, 2019: 4.10, 2020: 5.20, 2021: 8.10,
    2022: 5.90, 2023: 4.60, 2024: 4.30, 2025: 4.84, 2026: 4.17,
}

# Anos com quebra estrutural adicional (mudança legislativa relevante),
# além do reajuste ordinário do INPC.
QUEBRAS_LEGISLATIVAS = {
    2020: 0.06,   # Lei nº 11.400/2020 - redefinição de emolumentos/gratuidades
    2024: 0.04,   # Lei nº 12.325/2024 - inclusão do art. 28-A
}


def gerar_serie_sintetica(
    inicio: str = "2012-01-01",
    fim: str = "2025-12-01",
    nivel_base: float = 1_200_000.0,
    crescimento_real_aa: float = 0.03,
    amplitude_sazonal: float = 0.12,
    ruido_pct: float = 0.035,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera a série sintética mensal de arrecadação de emolumentos.

    Parameters
    ----------
    inicio, fim : datas (YYYY-MM-DD) delimitando a série mensal gerada.
    nivel_base : nível médio de arrecadação mensal no primeiro ano (R$).
    crescimento_real_aa : crescimento real anual do volume de atos (fração).
    amplitude_sazonal : amplitude da variação sazonal intra-anual (fração).
    ruido_pct : desvio padrão do ruído multiplicativo (fração).
    seed : semente do gerador aleatório (reprodutibilidade).

    Returns
    -------
    pd.DataFrame indexado por mês, com colunas:
        arrecadacao   -> valor sintético de arrecadação (R$)
        inpc_pct      -> reajuste do INPC aplicado naquele mês (0 fora de janeiro)
        reajuste_ano  -> reajuste acumulado do INPC vigente naquele ano (%)
        dummy_lei     -> 1 nos anos de quebra legislativa adicional, 0 caso contrário
        mes           -> mês do ano (1-12), útil para features sazonais
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(inicio, fim, freq="MS")

    fator_preco = 1.0
    fator_preco_serie = []
    inpc_pct_serie = []
    dummy_lei_serie = []

    ano_corrente = idx[0].year - 1
    for data in idx:
        if data.month == 1 and data.year in INPC_REAJUSTE:
            reajuste = INPC_REAJUSTE[data.year] / 100.0
            fator_preco *= (1 + reajuste)
            inpc_pct_serie.append(INPC_REAJUSTE[data.year])
        else:
            inpc_pct_serie.append(0.0)
        fator_preco_serie.append(fator_preco)
        dummy_lei_serie.append(1 if data.year in QUEBRAS_LEGISLATIVAS else 0)

    fator_preco_serie = np.array(fator_preco_serie)

    # Tendência real (efeito "quantidade": mais atos praticados ao longo do tempo)
    anos_decorridos = (idx.year - idx.year.min()) + (idx.month - 1) / 12.0
    fator_quantidade = (1 + crescimento_real_aa) ** anos_decorridos

    # Sazonalidade mensal estilizada (picos em dezembro/janeiro e maio; vale em fevereiro)
    sazonal_mensal = {
        1: 0.02, 2: -0.10, 3: 0.00, 4: 0.03, 5: 0.08, 6: 0.01,
        7: 0.00, 8: 0.02, 9: 0.01, 10: 0.03, 11: 0.05, 12: 0.10,
    }
    fator_sazonal = np.array([1 + sazonal_mensal[m] * (amplitude_sazonal / 0.12) for m in idx.month])

    # Quebra legislativa adicional: incremento permanente aplicado a partir do ano de vigência
    fator_legislativo = np.ones(len(idx))
    acumulado = 1.0
    for i, data in enumerate(idx):
        if data.month == 1 and data.year in QUEBRAS_LEGISLATIVAS:
            acumulado *= (1 + QUEBRAS_LEGISLATIVAS[data.year])
        fator_legislativo[i] = acumulado

    ruido = rng.normal(loc=1.0, scale=ruido_pct, size=len(idx))

    arrecadacao = (
        nivel_base
        * fator_quantidade
        * fator_preco_serie
        * fator_sazonal
        * fator_legislativo
        * ruido
    )

    df = pd.DataFrame(
        {
            "arrecadacao": arrecadacao,
            "inpc_pct": inpc_pct_serie,
            "dummy_lei": dummy_lei_serie,
            "mes": idx.month,
        },
        index=idx,
    )
    df.index.name = "competencia"
    return df


if __name__ == "__main__":
    df = gerar_serie_sintetica()
    print(df.head(15))
    print("...")
    print(df.tail(5))
    print(f"\nTotal de observações: {len(df)}")
