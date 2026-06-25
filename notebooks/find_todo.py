import json

with open(r'C:\ai_vinuni\code_vinuni\Day21-Track3-Finetuning-LLMs-LoRA-QLoRA\notebooks\Lab21_LoRA_Finetuning_T4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if 'TODO' in source or 'YÊU CẦU' in source:
        print(f"--- Cell {i} ({cell['cell_type']}) ---")
        print(source)
