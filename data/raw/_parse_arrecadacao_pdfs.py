import re, sys, json

def parse_month_sections(text):
    lines = text.split('\n')
    month_pattern = re.compile(r'Mês:\s*([A-ZÇÃÕÁÉÍÓÚÂÊÔÑa-zçãõáéíóúâêô]+/\d{4})')
    row_pattern = re.compile(r'\d{1,3}(?:\.\d{3})*,\d{2}')
    current_month = None
    totals = {}
    counts = {}
    for line in lines:
        m = month_pattern.search(line)
        if m:
            current_month = m.group(1).upper()
            totals.setdefault(current_month, 0.0)
            counts.setdefault(current_month, 0)
            continue
        if current_month and 'FERJ (B)' not in line and 'DESPESAS' not in line and 'Arrecadação' not in line and 'Bruta (A)' not in line:
            matches = row_pattern.findall(line)
            if matches:
                first = matches[0]
                val = float(first.replace('.', '').replace(',', '.'))
                totals[current_month] += val
                counts[current_month] += 1
    return totals, counts

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    totals, counts = parse_month_sections(text)
    for month in totals:
        print(f"{month:18s}  n={counts[month]:4d}  total_arrecadacao_bruta = R$ {totals[month]:,.2f}")
