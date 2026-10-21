# Previsão da Arrecadação de Emolumentos Extrajudiciais do TJMA com Redes Neurais Recorrentes

Pipeline experimental de comparação entre três arquiteturas de previsão de séries
temporais — **(1)** um modelo clássico (SARIMA/SARIMAX), **(2)** uma rede neural
recorrente (**LSTM**, implementada em NumPy puro) e **(3)** um modelo de *boosting*
sobre árvores (**XGBoost**) — aplicada à arrecadação mensal de emolumentos dos
serviços extrajudiciais (cartórios) do Tribunal de Justiça do Maranhão (TJMA).

Este repositório é a prova de conceito referenciada no **Capítulo 4 (Resultados
Preliminares)** do Projeto de Qualificação de Mestrado associado a esta pesquisa.

## ✅ Dados: série REAL obtida junto ao FERJ/TJMA

A pipeline já roda sobre a **série real de arrecadação bruta mensal** das
serventias extrajudiciais do Maranhão, consolidada a partir das publicações
mensais oficiais "Arrecadação Mensal das Serventias Extrajudiciais" (FERJ/TJMA)
e do arquivo histórico consolidado fornecido pela Corregedoria:

- **Período:** janeiro/2019 a setembro/2026 — **93 meses, sem lacunas**.
- **Nível de agregação:** total estadual (soma da Arrecadação Bruta (A) de
  ~280–290 serventias por mês), conforme cenário-base definido na Seção 3.1 do
  Projeto de Qualificação.
- **Como foi extraída:** os PDFs mensais publicados pelo FERJ foram processados
  com `pdftotext -layout` e um parser próprio (`data/raw/_parse_arrecadacao_pdfs.py`
  e `_consolidar_serie.py`) que soma a coluna "Arrecadação Bruta (A)" de cada
  serventia, mês a mês. O resultado consolidado está em
  `data/raw/arrecadacao_bruta_mensal_real_2019_2026.csv`.
- **Limitação conhecida:** o arquivo histórico também continha dados de
  2017–2018, mas dois meses (dezembro/2017 e dezembro/2018) apresentaram dois
  padrões de tabela distintos misturados no mesmo PDF, o que inflacionava
  artificialmente a soma; esses dois meses foram excluídos desta versão. Os
  demais meses de 2017–2018 já extraem corretamente e podem ser incorporados em
  uma próxima iteração, estendendo a série para além de 9 anos.

A série **sintética** originalmente usada como placeholder continua disponível
(`src/data/synthetic_generator.py`) e pode ser executada com
`python scripts/run_pipeline.py sintetica`, para fins de comparação metodológica.

## Estrutura do repositório

```
tjma-emolumentos-forecast/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── arrecadacao_bruta_mensal_real_2019_2026.csv   # série real consolidada
│   │   ├── _parse_arrecadacao_pdfs.py                     # extrator dos PDFs mensais
│   │   └── _consolidar_serie.py                           # consolidação em série única
│   └── processed/
│       ├── serie_real.csv
│       └── serie_sintetica.csv
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py   # gera a série sintética (referência/placeholder)
│   │   ├── real_data_loader.py      # carrega a série REAL consolidada
│   │   └── preprocessing.py         # lags, janelas, split temporal, normalização
│   ├── models/
│   │   ├── classical.py             # SARIMA/SARIMAX (baseline)
│   │   ├── lstm_numpy.py            # LSTM implementada em NumPy (forward+BPTT+Adam)
│   │   └── xgboost_model.py         # XGBoost sobre features de defasagem
│   ├── evaluation/
│   │   └── metrics.py               # MAE, RMSE, MAPE
│   └── visualization/
│       └── plots.py                 # geração dos 4 gráficos de saída
├── scripts/
│   └── run_pipeline.py              # ponto de entrada único (ver "Como executar")
├── tests/
│   ├── test_metrics.py
│   └── test_preprocessing.py
└── outputs/
    ├── figures/                     # gráficos PNG (com e sem sufixo _sintetico)
    └── results/                     # métricas e previsões em CSV
```

## Por que LSTM implementada em NumPy, e não PyTorch/TensorFlow?

Para manter o ambiente leve e sem dependência de bibliotecas de GPU/CUDA (que
somam vários gigabytes e não trazem benefício em CPU), a LSTM foi implementada
manualmente em NumPy puro — forward pass, *backpropagation through time* (BPTT)
e otimizador Adam, todos em `src/models/lstm_numpy.py`. Para a fase de
dissertação, com mais tempo disponível, se for pertinente, será reimplementado em PyTorch
ou TensorFlow/Keras, ganhando acesso a regularização (*dropout*), empilhamento
de camadas e aceleração por GPU — mudança que não deve alterar qualitativamente
os resultados obtidos aqui, dado o tamanho ainda moderado da série (93 meses).

## Como executar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/run_pipeline.py            # série REAL (padrão)
python3 scripts/run_pipeline.py sintetica  # série sintética (comparação)
```

O script imprime o progresso de cada etapa, treina as três arquiteturas, avalia
no período de teste (últimos 12 meses da série, out/2025–set/2026) e gera:

- `outputs/results/metricas_teste.csv` — MAE/RMSE/MAPE de cada modelo;
- `outputs/results/predicoes_teste.csv` — previsão mês a mês vs. valor real;
- `outputs/results/previsao_proximos_12_meses.csv` — previsão multi-step dos 12
  meses seguintes (out/2026–set/2027), caso de uso central do objetivo geral
  da pesquisa;
- `outputs/figures/*.png` — os 4 gráficos comparativos.

Tempo total de execução: cerca de 15 segundos em CPU comum (dados reais, 93
meses) — a maior parte gasta no treinamento da LSTM em NumPy.

## Resultados obtidos (dados REAIS, jan/2019–set/2026)

Avaliação *walk-forward* um-passo-à-frente no período de teste (12 meses mais
recentes: outubro/2025 a setembro/2026):

| Modelo | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| **XGBoost** | 4.384.294,86 | 5.835.451,46 | **8,20** |
| **ARIMA/SARIMA** | 4.237.753,15 | 4.745.362,49 | 8,36 |
| **LSTM (NumPy)** | 5.594.108,73 | 7.965.571,28 | 10,29 |

As três arquiteturas usam apenas a série de arrecadação (o XGBoost, também o
calendário, por meio de defasagens, média móvel e sazonalidade cíclica do mês),
sem variáveis externas. As diferenças de desempenho entre os modelos são
pequenas e devem ser lidas como indicativas, não como evidência de
superioridade de uma arquitetura sobre as demais.

O XGBoost e o modelo clássico (ARIMA/SARIMA) tiveram desempenho muito próximo e
superior ao da LSTM — resultado consistente com a **Hipótese H1** do
Projeto de Qualificação (Seção 1.5), que antecipava que, com volume de dados
ainda moderado (81 meses de treino), modelos clássicos ou de árvore tendem a
igualar ou superar redes neurais recorrentes. O MAPE em torno de 8-10% é um
patamar razoável para uma primeira rodada, sem otimização de hiperparâmetros —
há margem clara de melhoria a explorar na fase de dissertação (ajuste fino da
ordem do SARIMA, tamanho de janela e regularização da LSTM, profundidade do
XGBoost).

Os quatro gráficos gerados (`outputs/figures/`) mostram, respectivamente:
**(1)** a série real completa (2019–2026) com o corte treino/teste, incluindo
visivelmente a queda associada à pandemia em 2020; **(2)** a previsão
um-passo-à-frente das três arquiteturas contra o valor real no período de
teste; **(3)** a comparação de métricas de erro; e **(4)** a previsão
multi-step dos 12 meses seguintes (out/2026–set/2027).

## Próximos passos (fase de dissertação)

1. Obter a série em nível mais desagregado (por comarca ou tipo de ato) e
   reconciliar dezembro/2017 e dezembro/2018 para estender a série a ~9,3 anos.
2. Ajuste fino de hiperparâmetros das três arquiteturas (busca em grade ou
   aleatória), hoje usando configurações razoáveis mas não otimizadas.
3. Reimplementar a LSTM em PyTorch/TensorFlow para viabilizar regularização e
   arquiteturas mais profundas (GRU, LSTM empilhada).

## Testes

```bash
python3 tests/test_metrics.py
python3 tests/test_preprocessing.py
```

## Referência

Este repositório implementa a metodologia descrita no Projeto de Qualificação de
Mestrado *"Previsão da Arrecadação de Emolumentos Extrajudiciais do Tribunal de
Justiça do Maranhão: uma Análise Comparativa de Arquiteturas de Aprendizado de
Máquina e Séries Temporais"* (Capítulo 3 — Metodologia).
