# Builds the fixtures listed in fixtures.txt ("<name> <path under Dynare's tests/>"): the test file cut before its first line that needs
# dseries, a simulation, plotting/printing or an estimation (JDyn runs the part before; Octave produces the references of gen_refs.sh).
import re, sys
tests = sys.argv[1] if len(sys.argv) > 1 else '/root/ref/dynare-7.2/tests'
for line in open('fixtures.txt'):
    name, path = line.split()
    lines = open(f'{tests}/{path}').read().split('\n')
    n = next((i for i, l in enumerate(lines) if re.search(r'dseries|initialconditions|simul_backward|search\(|cherrypick|pac\.estimate|pac\.print|var_expectation\.print', l)), None)
    txt = '\n'.join(lines[:n] if n is not None else lines) + '\n'
    open(name + '.mod', 'w').write(txt.replace('example1.varmatrices', name + '.varmatrices'))      # the package of generated functions is named after the file
