import sys, shutil, warnings, time, numpy as np, tempfile
import pydyn
from pydyn.obc_ho import HigherOrderNews
from scipy.io import loadmat
REF='/home/claude/pydyn/reference/dynareobc_octave/'
d=tempfile.mkdtemp(); shutil.copy(REF+'nk_occ.mod', d)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = pydyn.Model(pydyn.run_preprocessor(d+'/nk_occ.mod'))
S = loadmat(REF+'shocks.mat',squeeze_me=True)['ShockSequence'].reshape(-1,1)
for lbl,kw in (("order1",dict(order=1)),("order2",dict(order=2)),("cub9",dict(order=2,cubature=True,qmc_level=9)),("cub10",dict(order=2,cubature=True,qmc_level=10))):
    h=HigherOrderNews(m,horizon=32,check_ahead=32,**kw); h.simulate(S[:20])
    ts=[]
    for _ in range(5):
        t=time.perf_counter(); h.simulate(S); ts.append((time.perf_counter()-t)/300*1e6)
    print(lbl, round(min(ts),1), round(sorted(ts)[2],1))
