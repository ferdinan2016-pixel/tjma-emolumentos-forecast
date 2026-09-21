# Previsão da Arrecadação de Emolumentos Extrajudiciais do TJMA com Redes Neurais Recorrentes

Pipeline experimental de comparação entre três arquiteturas de previsão de séries
temporais — **(1)** um modelo clássico (SARIMA/SARIMAX), **(2)** uma rede neural
recorrente (**LSTM**) e **(3)** um modelo de *boosting* sobre árvores (**XGBoost**) —
aplicada à arrecadação mensal de emolumentos dos serviços extrajudiciais (cartórios)
do Tribunal de Justiça do Maranhão (TJMA).

Este repositório é a prova de conceito referenciada no **Capítulo 4 (Resultados
Preliminares)** do Projeto de Qualificação de Mestrado associado a esta pesquisa.

## ⚠️ Aviso importante sobre os dados

A série histórica **real** de arrecadação ainda não foi obtida junto ao FERJ/Corregedoria
do Foro Extrajudicial do TJMA (etapa 2 do cronograma do Projeto de Qualificação). Por
isso, **todos os resultados numéricos deste repositório foram gerados sobre uma série
SINTÉTICA** (`src/data/synthetic_generator.py`), construída para reproduzir, de forma
estilizada, a estrutura descrita no marco jurídico da pesquisa:

- tendência de crescimento real do volume de atos praticados;
- sazonalidade mensal;
- reajuste anual da tabela de emolumentos pelo INPC, aplicado em janeiro (os
  percentuais de 2025 e 2026 usam os valores efetivamente noticiados pelo TJMA —
  4,84% e 4,17% — os demais anos são simulados dentro da faixa histórica do índice);
- duas quebras estruturais adicionais associadas às Leis nº 11.400/2020 e nº 12.325/2024;
- ruído aleatório.

**Nenhum valor de arrecadação aqui apresentado corresponde à arrecadação real do TJMA.**
O objetivo deste repositório é exclusivamente demonstrar que a pipeline completa —
coleta/carga → pré-processamento → treinamento das três arquiteturas → avaliação →
geração de gráficos — está implementada, funcional e reprodutível de ponta a ponta,
antes da chegada dos dados reais. Ao obtê-los, basta substituir a função
`gerar_serie_sintetica()` por uma função de carga da série real (ver seção
"Como migrar para dados reais" abaixo); o restante do pipeline não muda.

## Estrutura do repositório

```
tjma-emolumentos-forecast/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                       # (vazio) destino da série real, quando obtida
│   └── processed/
│       └── serie_sintetica.csv    # série sintética gerada pelo pipeline
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py # gera a série sintética (placeholder)
│   │   └── preprocessing.py       # lags, janelas, split temporal, normalização
│   ├── models/
│   │   ├── classical.py           # SARIMA/SARIMAX (baseline)
│   │   ├── lstm_numpy.py          # LSTM implementada em NumPy (forward+BPTT+Adam)
│   │   └── xgboost_model.py       # XGBoost sobre features de defasagem
│   ├── evaluation/
│   │   └── metrics.py             # MAE, RMSE, MAPE
│   └── visualization/
│       └── plots.py               # geração dos 4 gráficos de saída
├── scripts/
│   └── run_pipeline.py            # ponto de entrada único (ver "Como executar")
├── tests/
│   ├── test_metrics.py
│   └── test_preprocessing.py
└── outputs/
    ├── figures/                   # 4 gráficos PNG gerados pelo pipeline
    └── results/                   # métricas e previsões em CSV
```

Essa organização segue a Seção 3.5 (Engenharia de software da solução) do Projeto de
Qualificação: separação por responsabilidade (dados, modelos, avaliação, visualização),
scripts de orquestração, testes automatizados e saídas versionáveis.

## Por que LSTM implementada em NumPy, e não PyTorch/TensorFlow?

Para manter o ambiente de prova de conceito leve e sem dependência de bibliotecas de
GPU/CUDA (que somam vários gigabytes e não trazem benefício em CPU), a LSTM foi
implementada manualmente em NumPy puro — forward pass, *backpropagation through time*
(BPTT) e otimizador Adam, todos em `src/models/lstm_numpy.py`. Essa decisão é
documentada como escolha deliberada de engenharia: para a fase de dissertação, com a
base real e maior volume de dados, recomenda-se reimplementar em PyTorch ou
TensorFlow/Keras, ganhando acesso a regularização (*dropout*), empilhamento de camadas
e aceleração por GPU.

## Como executar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/run_pipeline.py
```

O script imprime o progresso de cada etapa, treina as três arquiteturas, avalia no
período de teste (últimos 12 meses da série) e gera:

- `outputs/results/metricas_teste_2025.csv` — MAE/RMSE/MAPE de cada modelo;
- `outputs/results/predicoes_teste_2025.csv` — previsão mês a mês vs. valor real;
- `outputs/results/previsao_2026.csv` — previsão multi-step dos 12 meses seguintes
  ("arrecadação do ano seguinte", caso de uso central do objetivo geral da pesquisa);
- `outputs/figures/*.png` — os 4 gráficos comparativos.

Tempo total de execução: cerca de 20 segundos em CPU comum (a maior parte gasta no
treinamento da LSTM em NumPy, ~19s para 150 épocas).

## Resultados obtidos (sobre a série sintética)

Avaliação *walk-forward* um-passo-à-frente no período de teste (12 meses, equivalente
a um ano de "2025"):

| Modelo | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| **XGBoost** | 161.077,94 | 201.615,68 | **3,77** |
| **ARIMA/SARIMA** | 162.648,37 | 185.835,74 | 3,81 |
| **LSTM (NumPy)** | 211.272,93 | 246.193,27 | 5,09 |

Nesta prova de conceito, o modelo clássico (ARIMA/SARIMA) e o XGBoost superaram
levemente a LSTM — resultado consistente com a **Hipótese H1** do Projeto de
Qualificação (Seção 1.5), que antecipava que, em séries curtas/com volume de dados
reduzido, os modelos clássicos ou de árvore tendem a igualar ou superar redes
neurais recorrentes. Esse é exatamente o tipo de achado que a pipeline foi desenhada
para poder produzir — e que só poderá ser confirmado ou refutado com a série real.

Previsão multi-step para os 12 meses seguintes ("ano seguinte"), totais anuais por
modelo (valores fictícios, sobre a série sintética):

| Modelo | Total anual previsto |
|---|---:|
| ARIMA/SARIMA | R$ 54.996.933,19 |
| LSTM | R$ 59.194.167,21 |
| XGBoost | R$ 50.627.897,50 |

Os quatro gráficos gerados (`outputs/figures/`) mostram, respectivamente: **(1)** a
série sintética completa com o corte treino/teste; **(2)** a previsão um-passo-à-frente
das três arquiteturas contra o valor real no período de teste; **(3)** a comparação de
métricas de erro; e **(4)** a previsão multi-step do ano seguinte.

## Como migrar para dados reais

1. Obter a série de arrecadação junto ao FERJ/Corregedoria do TJMA e salvá-la em
   `data/raw/` (formato sugerido: CSV com colunas `competencia` [mês] e `arrecadacao`).
2. Escrever uma função `carregar_serie_ferj(caminho_csv)` em
   `src/data/` que leia esse CSV e devolva um `DataFrame` no mesmo formato produzido
   por `gerar_serie_sintetica()` (índice mensal, coluna `arrecadacao`, e — se
   disponíveis — `inpc_pct`/`dummy_lei`).
3. Trocar, em `scripts/run_pipeline.py`, a chamada
   `df = gerar_serie_sintetica()` por `df = carregar_serie_ferj("data/raw/serie_ferj.csv")`.
4. Reexecutar `python3 scripts/run_pipeline.py` — todo o restante da pipeline
   (pré-processamento, treino das três arquiteturas, avaliação, gráficos) permanece
   inalterado.
5. Reavaliar hiperparâmetros (ordem do SARIMA, tamanho da janela e nº de unidades da
   LSTM, profundidade do XGBoost) em função do volume e da granularidade reais dos
   dados obtidos, conforme discutido na Seção 3.6 do Projeto de Qualificação.

## Testes

```bash
python3 tests/test_metrics.py
python3 tests/test_preprocessing.py
```

## Referência

Este repositório implementa a metodologia descrita no Projeto de Qualificação de
Mestrado *"Previsão da Arrecadação de Emolumentos Extrajudiciais do Tribunal de
Justiça do Maranhão mediante Redes Neurais Recorrentes: uma Análise Comparativa de
Arquiteturas"* (Capítulo 3 — Metodologia).
