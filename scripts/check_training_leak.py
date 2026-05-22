import re
import sys
import os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FORBIDDEN_PATTERNS = [
    re.compile(r'nn\.CrossEntropyLoss'),
    re.compile(r'nn\.MSELoss'),
    re.compile(r'nn\.KLDivLoss'),
    re.compile(r'torch\.optim\.'),
    re.compile(r'\.backward\(\)'),
    re.compile(r'optimizer\.step\(\)'),
    re.compile(r'lr_scheduler'),
    re.compile(r'learning_rate\s*=\s*\d'),
    re.compile(r'stage1|stage2|stage3|stage4'),
    re.compile(r'四阶段'),
    re.compile(r'训练配方'),
    re.compile(r'数据配方'),
]

EXCLUDE_FILES = {
    'train_adapter.py',
    'train_physics_adapter.py',
    '_create_release.py',
    'check_training_leak.py',
    'eval_benchmark.py',
    'build_tokenizer.py',
    'app.py',
    'app_spike.py',
    'demo_real_text.py',
    'spike_interface.py',
}

def check_file(filepath):
    basename = os.path.basename(filepath)
    if basename in EXCLUDE_FILES:
        return []
    if not filepath.endswith('.py'):
        return []

    findings = []
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                for pat in FORBIDDEN_PATTERNS:
                    if pat.search(line):
                        findings.append((filepath, i, line.strip()[:80]))
                        break
    except Exception:
        pass
    return findings

def main():
    all_findings = []
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            path = os.path.join(root, f)
            all_findings.extend(check_file(path))

    if all_findings:
        print("[WARNING] Potential training logic detected in the following files:")
        print("    (If this is intentional, add the filename to EXCLUDE_FILES)")
        print()
        for filepath, lineno, line in all_findings:
            print(f"  {filepath}:{lineno}")
            print(f"    {line}")
            print()
        sys.exit(1)
    else:
        print("[OK] No training logic leaks detected")
        sys.exit(0)

if __name__ == '__main__':
    main()
