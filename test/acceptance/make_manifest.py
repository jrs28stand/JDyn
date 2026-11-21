#!/usr/bin/env python3
"""Generate manifest.tsv from Dynare 7.2's meson.build (the list `mod_and_m_tests`).
Usage: python3 make_manifest.py /root/ref/dynare-7.2 > manifest.tsv
One line per meson entry (tests of one entry run in order in one scratch directory):
id <TAB> tests (;-separated) <TAB> extras (;-separated) <TAB> should_fail <TAB> dynare_args (;-separated) <TAB> env (k=v;...)"""
import re, sys
root = sys.argv[1]
txt = open(root + "/meson.build").read()
start = txt.index("mod_and_m_tests = [")
i = txt.index("[", start); depth = 0
for j in range(i, len(txt)):
    c = txt[j]
    if c == "[": depth += 1
    elif c == "]":
        depth -= 1
        if depth == 0: break
body = txt[i:j+1]
body = re.sub(r"#[^\n]*", "", body)                       # comments
body = body.replace("true", "True").replace("false", "False")
body = re.sub(r"\(get_option\([^)]*\)[^)]*\)\.to_string\(\)", "'x'", body)
body = re.sub(r"\(get_option\([^)]*\)\.enabled\(\)\)\.to_string\(\)", "'x'", body)
body = re.sub(r"get_option\([^)]*\)\.enabled\(\)\.to_string\(\)", "'x'", body)
entries = eval(body)
print("#id\ttests\textras\tshould_fail\tdynare_args\tenv")
for e in entries:
    t = e["test"]
    print("\t".join([t[-1], ";".join(t), ";".join(e.get("extra", [])), "1" if e.get("should_fail") else "0",
                     ";".join(e.get("dynare_args", [])), ";".join(f"{k}={v}" for k, v in e.get("env", {}).items())]))
