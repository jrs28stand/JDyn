using Test
include("/home/claude/work/jdyn-ho/test/common.jl")
const HO_DIR="/home/claude/work/jdyn-ho/test/obc/ho"; ho_ref(n)=matread(joinpath(HO_DIR,n))
m=load_copy(joinpath(HO_DIR,"nk_occ.mod")); ob=JDyn.Occbin(m)
S=reshape(vec(ho_ref("shocks.mat")["ShockSequence"]),:,1); names=String.(m.endo_names)
function ref_endo(f); r=ho_ref(f); nm=[strip(String(x)) for x in vec(r["names"])]; r,permutedims(r["endo"][[findfirst(==(v),nm) for v in names],:]); end
r,_=ref_endo("ref_o2cub.mat"); _,E4=ref_endo("ref_o2cub4.mat")
h=JDyn.obc_ho(ob;order=2,horizon=32,check_ahead=32,cubature=true,qmc_level=9,regions=4)
h.R,h.xi=real.(r["Rcc"]),ho_ref("sob_o2cub.mat")["x"]
t=@elapsed sim=JDyn.obc_ho_simulate(h,S)
d=vec(maximum(abs.(sim.with_bound.-E4);dims=2))
println(count(<(1e-12),d)," agree <1e-12; count<3e-16: ",count(<(3e-16),d)," max ",maximum(d)," time ",t)
