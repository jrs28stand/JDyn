import sys, shutil, tempfile, warnings, time
sys.path.insert(0,'/home/claude/pydyn')
import numpy as np, pydyn
from pathlib import Path
from scipy.io import loadmat
from pydyn.occbin_estimation import OccbinEstimation
warnings.simplefilter("ignore")
tmp=Path(tempfile.mkdtemp())
src=Path('/home/claude/ref/dynare-7.2/tests/estimation/slice/bf.mod'); shutil.copy(src,tmp/'bf.mod')
m=pydyn.Model(pydyn.run_preprocessor(tmp/'bf.mod'))
d=loadmat('/home/claude/pydyn/reference/occest_bf_data.mat',squeeze_me=True); data=np.column_stack([d["q"],d["r"]])
kw=dict(posterior_sampling_method="random_walk_metropolis_hastings", posterior_sampler_options=())
def est(**k): return OccbinEstimation(m,data={v:data[:,j] for j,v in enumerate(m.varobs)},nobs=data.shape[0],first_obs=1,occbin={"likelihood_inversion_filter":True},**kw,**k)
e0=est(); x0=np.array(e0.x0); print("x0",repr(x0))
ths=[x0*np.exp(0.05*np.sin(1.3*k*np.arange(1,6)+k)) for k in range(6)]
for sel in ("occbin","shortest","longest","min_wedge"):
    e=est(obc_method="news",obc_selection=sel)
    print(sel,[repr(e.log_posterior(t)) for t in ths])
print("occ",[repr(e0.log_posterior(t)) for t in ths])
