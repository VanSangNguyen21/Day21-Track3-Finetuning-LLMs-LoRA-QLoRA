import json

with open(r'C:\ai_vinuni\code_vinuni\Day21-Track3-Finetuning-LLMs-LoRA-QLoRA\notebooks\Lab21_LoRA_Finetuning_T4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(r'C:\ai_vinuni\code_vinuni\Day21-Track3-Finetuning-LLMs-LoRA-QLoRA\notebooks\notebook_preview.txt', 'w', encoding='utf-8') as f_out:
    for i, cell in enumerate(nb['cells'][:30]):
        f_out.write(f"\n\n=== Cell {i} ({cell['cell_type']}) ===\n")
        f_out.write("".join(cell.get('source', []))[:500])
