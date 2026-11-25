import sys, time, shutil, tempfile, warnings
sys.path.insert(0,'/home/claude/pydyn'); sys.path.insert(0,'/home/claude/pydyn/tests')
import numpy as np, pydyn
from pathlib import Path
from scipy.io import loadmat
from pydyn.occbin_estimation import OccbinEstimation
REF=Path('/home/claude/pydyn/reference'); tmp=Path(tempfile.mkdtemp())
warnings.simplefilter("ignore")
for case,n in (("nkms",3),("nkm",4)):
    shutil.copy(REF/f"occest_{case}.mod", tmp/f"occest_{case}.mod")
    m=pydyn.Model(pydyn.run_preprocessor(tmp/f"occest_{case}.mod"))
    r=loadmat(REF/f"occest_{case}_lik.mat",squeeze_me=True); data=np.atleast_2d(r["data"].T).T; X=np.atleast_2d(r["X"])
    def est(**kw): return OccbinEstimation(m,data={v:data[:,j] for j,v in enumerate(m.varobs)},nobs=data.shape[0],first_obs=1,occbin={"likelihood_inversion_filter":True},**kw)
    e0=est(); e1=est(obc_method="news"); X = X if X.shape[0]==e0.n else X.T
    for k in range(n): print(case,k+1,repr(e1.log_posterior(X[:,k])))
    th=X[:,1]; e0.log_posterior(th); e1.log_posterior(th)
    t0=min((lambda t:(e0.log_posterior(th),time.time()-t)[1])(time.time()) for _ in range(3))
    t1=min((lambda t:(e1.log_posterior(th),time.time()-t)[1])(time.time()) for _ in range(3))
    print(case,"pydyn ms occbin",1000*t0,"news",1000*t1)
