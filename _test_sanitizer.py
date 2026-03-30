theimport re

def _sanitize_process_id(pid: str) -> str:
    if not pid:
        return pid
    # Try to find the module code pattern (2-3 uppercase letters + dot + 2 digits)
    match = re.search(r'([A-Z]{2,3}\.\d{2})(?:\s|$|\.)', pid)
    if match:
        return match.group(1)
    # Fallback: dash-separated step_id format
    match = re.search(r'^([A-Z]{2,3})-(\d{2})(?:-\d+)?$', pid.upper().strip())
    if match:
        return f'{match.group(1)}.{match.group(2)}'
    return pid

# Test cases from the error log
cases = [
    'AL Gosaibi Co..AP.08',
    'AL.Gosaibi Co.AP.09',
    'AL Gosaibi Co.AP.01',
]

print("Testing _sanitize_process_id:")
for pid in cases:
    result = _sanitize_process_id(pid)
    expected = pid.split('.')[-1] if '.' in pid else pid
    # Extract AP.XX pattern
    m = re.search(r'([A-Z]{2,3})\.(\d{2})', pid)
    expected = f"{m.group(1)}.{m.group(2)}" if m else "???"
    status = "OK" if result == expected else "FAIL"
    print(f'  [{status}] {pid!r:32} → {result!r:10} (expected {expected!r})')
