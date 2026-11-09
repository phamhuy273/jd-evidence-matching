import glob
import re

print("=== CHECKING ALL TEX FILES FOR FORMATTING ISSUES ===")
tex_files = glob.glob('paper/**/*.tex', recursive=True) + glob.glob('paper/*.tex')
tex_files = sorted(list(set(tex_files)))

for fpath in tex_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        uncommented = line.split('%')[0]
        # check unescaped dollar count
        dollars = re.findall(r'(?<!\\)\$', uncommented)
        if len(dollars) % 2 != 0:
            print(f"[ODD $ COUNT] {fpath}:{idx+1}: {line.strip()}")
        # check words inside math mode
        matches = re.findall(r'\$([^$]+)\$', uncommented)
        for m in matches:
            # If contains words with more than 3 letters that are not standard math macros
            words = re.findall(r'[A-Za-z]{3,}', m)
            bad_words = [w for w in words if w not in [
                'text', 'mathrm', 'mathbf', 'mathit', 'textbf', 'textit', 'emph',
                'min', 'max', 'log', 'exp', 'sin', 'cos', 'cdot', 'times',
                'alpha', 'beta', 'gamma', 'delta', 'kappa', 'sigma', 'tau',
                'Delta', 'Gamma', 'Lambda', 'Sigma', 'Omega',
                'approx', 'le', 'ge', 'ne', 'in', 'notin', 'dag', 'dagger',
                'astH', 'astN', 'lineH', 'lineN', 'castname', 'allowbreak',
                'quad', 'qquad', 'to', 'left', 'right', 'pm', 'circ'
            ]]
            if bad_words:
                print(f"[UNEXPECTED WORDS IN MATH] {fpath}:{idx+1}: bad={bad_words} in ${m}$")

print("=== SCAN COMPLETE ===")
