import urllib.request, json
req = urllib.request.Request('https://openrouter.ai/api/v1/models')
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    
models = data.get('data', [])

# Broaden the search
families = ['llama', 'claude', 'gpt', 'deepseek', 'gemini', 'qwen', 'mixtral', 'phi']

results = []
for m in models:
    name = m['id'].lower()
    ctx = m.get('context_length', 0)
    
    # Only include models with at least 16k context, matching families
    if ctx >= 16000 and any(f in name for f in families):
        cost_p = m.get('pricing', {}).get('prompt', '0')
        cost_c = m.get('pricing', {}).get('completion', '0')
        try:
            p = float(cost_p) * 1_000_000
            c = float(cost_c) * 1_000_000
        except:
            p, c = 0, 0
            
        results.append({
            'id': m['id'],
            'ctx': ctx,
            'prompt_1M': f'${p:.2f}',
            'comp_1M': f'${c:.2f}',
            'sort_val': p
        })

# Sort by input price
results.sort(key=lambda x: (x['sort_val'], x['id']))

print(f"{'Model':<50} | {'Context':<8} | {'In/1M':<7} | {'Out/1M'}")
print("-" * 80)
for r in results[:60]:
    print(f"{r['id']:<50} | {r['ctx']:<8} | {r['prompt_1M']:<7} | {r['comp_1M']}")
