import shutil,tempfile,warnings,numpy as np, pydyn
from pydyn.obc import NewsOBC
d=tempfile.mkdtemp(); shutil.copy('/home/claude/pydyn/reference/dynareobc_octave/simple1_occ.mod', d)
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); m=pydyn.Model(pydyn.run_preprocessor(d+'/simple1_occ.mod'))
nb=NewsOBC(m,horizon=40,check_ahead=40)
try:
    r=nb.simulate(np.array([[-0.02]]),periods=20); print('py ok', np.abs(r.piecewise).max())
except Exception as e: print('py', type(e).__name__, str(e)[:200])
