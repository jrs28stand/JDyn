import sys, shutil, warnings, numpy as np
sys.path.insert(0,'/home/claude/pydyn')
import pydyn
from pydyn.obc_ho import HigherOrderNews
from scipy.io import loadmat
REF='/home/claude/pydyn/reference/dynareobc_octave/'
import tempfile, os
d=tempfile.mkdtemp(); shutil.copy(REF+'nk_occ.mod', d)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = pydyn.Model(pydyn.run_preprocessor(d+'/nk_occ.mod'))
h = HigherOrderNews(m, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=9)
Rd = np.real(loadmat(REF+'ref_o2cub.mat',squeeze_me=True)['Rcc'])
print([int(np.sign(h.R[:,j]@Rd[:,j])) for j in range(12)])
S = loadmat(REF+'shocks.mat',squeeze_me=True)['ShockSequence'].reshape(-1,1)
r = loadmat(REF+'ref_o2cub.mat',squeeze_me=True)
names=[str(n).strip() for n in r['names']]
E = r['endo'][[names.index(n) for n in m.endo_names]].T
own = h.simulate(S)['with_bound']
print('python own err', np.abs(own-E).max())
np.save('/tmp/ho/py_R.npy', h.R)
