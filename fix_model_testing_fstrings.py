# fix_model_testing_fstrings.py
# Usage: py -3.11 fix_model_testing_fstrings.py
from pathlib import Path
import re
import shutil
import sys

SRC = Path(r"C:\Users\nvenkata\Downloads\daddy\daddy\model_testing.py")
if not SRC.exists():
    print("ERROR: source file not found:", SRC)
    sys.exit(1)

# backup original
bak = SRC.with_suffix(".py.bak")
shutil.copy2(SRC, bak)
print("Backup created at:", bak)

text = SRC.read_text(encoding="utf-8")

# 1) Fix the very common pattern that caused your SyntaxError:
#    st.write(f"- Contains dates: {bool(re.search(r'\\b(19|20)\\d{2}\\b', resume_text))}")
# -> replace with:
#    contains_dates = bool(re.search(r"\b(19|20)\d{2}\b", resume_text))
#    st.write(f"- Contains dates: {contains_dates}")

# General approach:
# - Find st.write(...) lines that contain an f-string and a re.search(...) inside {...}
# - Replace by computing the boolean into a temporary variable before the st.write call
# - Use a generated variable name (unique) to avoid name clashes

pattern_fsearch = re.compile(
    r'(^\s*)(st\.write\()\s*(f[\'"])(?P<body>.*?re\.search\(\s*r(?P<q1>[\'"])(?P<pat>.*?)(?P=q1)\s*,\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*\).*?)([\'"]\)\s*)$',
    re.DOTALL | re.MULTILINE
)

counter = 0
def repl(match):
    global counter
    indent = match.group(1)
    body = match.group('body')
    pat = match.group('pat')
    var = match.group('var')

    # sanitize pattern for embedding in a python string literal (double-quoted)
    pat_escaped = pat.replace('"', r'\"')
    tmp_var = f"__fixed_re_bool_{counter}"
    counter_inc = None
    # Build replacement lines: compute bool before, then write original message replacing the re.search(...) with the tmp var
    # We will replace the occurrence of the re.search(...) inside the body with {tmp_var} if body uses {} expression
    # To keep it simple we will not attempt to reconstruct the whole f-string body; instead we will craft a safe st.write line
    # that approximates the original intent:
    #   st.write("- Contains dates: " + str(bool(re.search(r"...", resume_text))))
    # So we'll format as concatenation with str(bool(re.search(...)))
    new_line = (f"{indent}{tmp_var} = bool(re.search(r\"{pat_escaped}\", {var}))\n"
                f"{indent}st.write(\"- Contains dates: \" + str({tmp_var}))")
    # increment counter
    counter_inc = 1
    globals()['counter'] += counter_inc
    return new_line

# Apply replacement iteratively for matches
new_text, nsubs = pattern_fsearch.subn(repl, text)
print(f"Applied pattern_fsearch substitutions: {nsubs}")

# If no substitutions found by the general pattern, handle a known specific line (fallback)
if nsubs == 0:
    # Known problematic line from earlier message: exact replace
    old = r"st.write(f\"- Contains dates: {bool(re.search(r'\\b(19|20)\\d{2}\\b', resume_text))}\")"
    if old in text:
        new_text = text.replace(
            old,
            'contains_dates = bool(re.search(r"\\b(19|20)\\d{2}\\b", resume_text))\n            st.write(f"- Contains dates: {contains_dates}")'
        )
        print("Applied fallback exact replacement for dates line.")
    else:
        print("No exact fallback pattern found; attempting broader search...")

# Broader pass: find any occurrence of an f-string with re.search containing a \\b and replace the {...} expression with concatenation
# This will catch other lines like: st.write(f"...{bool(re.search(r'\\b...\\b', resume_text))}...")
broad_pattern = re.compile(r'(st\.write\()\s*(f[\'"])(?P<body>.*?)([\'"]\))', re.DOTALL)
def broad_repl(m):
    body = m.group('body')
    if "re.search" in body and r"\b" in body:
        # find inner re.search call
        rs = re.search(r're\.search\(\s*r(?P<q>[\'"])(?P<pat>.*?)(?P=q)\s*,\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*\)', body)
        if rs:
            pat = rs.group('pat').replace('"', r'\"')
            var = rs.group('var')
            return f'st.write("- Contains dates: " + str(bool(re.search(r"{pat}", {var}))))'
    return m.group(0)

new_text2 = broad_pattern.sub(broad_repl, new_text)
if new_text2 != new_text:
    print("Applied broad pass replacements.")
    new_text = new_text2

# Write fixed file
out = SRC.with_name("model_testing_fixed.py")
out.write_text(new_text, encoding="utf-8")
print("Fixed file written to:", out)
print("Original backed up at:", bak)
