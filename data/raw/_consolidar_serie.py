import re
from parse_arrecadacao import parse_month_sections
import pandas as pd

MESES = {
    "JANEIRO": 1, "FEVEREIRO": 2, "MARÇO": 3, "ABRIL": 4, "MAIO": 5, "JUNHO": 6,
    "JULHO": 7, "AGOSTO": 8, "SETEMBRO": 9, "OUTUBRO": 10, "NOVEMBRO": 11, "DEZEMBRO": 12,
}

def load(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return parse_month_sections(text)

registros = {}  # (ano, mes) -> (total, n)

# 1) histórico 2019-2024 (do arquivo grande) -- pegar SOMENTE 2019-2024, ignorando 2017/2018 por enquanto
totals, counts = load("hist_2019_2024.txt")
for label, total in totals.items():
    nome, ano = label.split("/")
    ano = int(ano)
    if ano < 2019 or ano > 2024:
        continue
    mes = MESES[nome.upper()]
    registros[(ano, mes)] = (total, counts[label])

# 2) 2025 completo
totals, counts = load("jan_dez_2025.txt")
for label, total in totals.items():
    nome, ano = label.split("/")
    ano = int(ano)
    mes = MESES[nome.upper()]
    registros[(ano, mes)] = (total, counts[label])

# 3) 2026 jan-jul (arquivos individuais)
arquivos_2026 = {
    1: "jan2026.txt", 2: "fev2026.txt", 3: "mar2026.txt", 4: "abr2026.txt",
    5: "mai2026.txt", 6: "jun2026.txt", 7: "jul2026.txt",
}
for mes, path in arquivos_2026.items():
    totals, counts = load(path)
    # so ha um mes por arquivo
    label = list(totals.keys())[0]
    registros[(2026, mes)] = (totals[label], counts[label])

# monta serie ordenada
linhas = []
for (ano, mes), (total, n) in sorted(registros.items()):
    linhas.append({"ano": ano, "mes": mes, "competencia": f"{ano}-{mes:02d}-01", "arrecadacao_bruta": total, "n_serventias": n})

df = pd.DataFrame(linhas)
df = df.sort_values("competencia").reset_index(drop=True)
print(df.to_string())
print(f"\nTotal de meses: {len(df)}")
print(f"Periodo: {df['competencia'].iloc[0]} a {df['competencia'].iloc[-1]}")
df.to_csv("serie_real_arrecadacao_bruta_mensal.csv", index=False)
