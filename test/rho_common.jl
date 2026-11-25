# Helpers of test/test_obc_regime_ho.jl and bench/obc_regime_ho_*.jl: the models of pydyn's tests/test_obc_regime_ho.py, their shock draws and constructors.
# (Needs `Test`, `common.jl` and `Random` loaded: `using Test; include("common.jl"); using Random`.)
const RHO_DIR = joinpath(@__DIR__, "obc", "ho")
const RHO_OBC = joinpath(@__DIR__, "obc")
const RHO_EST = joinpath(RHO_OBC, "est")
rho_ref(name) = matread(joinpath(RHO_DIR, name))
const RHO_SRC = Dict("nkm" => joinpath(@__DIR__, "ref_occbin", "occest_nkm.mod"), "zlb" => joinpath(RHO_OBC, "obc_zlb.mod"), "zlb2" => joinpath(RHO_OBC, "obc_zlb2.mod"),
                     "nk" => joinpath(RHO_DIR, "nk_occ.mod"), "borrcon" => joinpath(RHO_DIR, "occ_borrcon.mod"), "sw" => joinpath(RHO_OBC, "obc_sw_zlb.mod"),
                     "bf" => joinpath(RHO_EST, "bf.mod"), "zlb2m_0" => joinpath(RHO_EST, "zlb2m_0.mod"), "zlb2m_1e_2" => joinpath(RHO_EST, "zlb2m_1e_2.mod"))
# (shock scale relative to the model's sd, horizon) chosen so that the constraint binds in a good share of 150 periods (pydyn's CFG)
const RHO_CFG = Dict("nkm" => (1.5, 40), "zlb" => (0.02, 20), "zlb2" => (0.02, 20), "nk" => (1.0, 32), "borrcon" => (0.5, 40), "sw" => (0.4, 32), "bf" => (1.0, 32),
                     "zlb2m_0" => (1.0, 40), "zlb2m_1e_2" => (1.0, 40))
const _RHO_CACHE = Dict{String,Any}()
rho_model(name) = get!(() -> load_copy(RHO_SRC[name]; output="second"), _RHO_CACHE, name)
function rho_shocks(name; T=150, seed=4)
    m = rho_model(name)
    sc, _ = RHO_CFG[name]
    sd = name in ("zlb", "zlb2") ? ones(m.n_exo) : sqrt.(diag(Matrix(m.Sigma_e)))
    randn(MersenneTwister(seed), T, m.n_exo) .* sd' .* sc
end
"Shocks of zlb2 under which the strict rule fails (a class (b) point at period 2 with the full average, level 7): the failure tests."
rho_fail_shocks(; T=40) = randn(MersenneTwister(9), T, rho_model("zlb2").n_exo) .* 0.06
rho_binds(r) = count(w -> maximum(abs, w) > 1e-10, eachrow(r.wedges))
quiet(f) = with_logger(f, NullLogger())
mkreg(name; kw...) = (m = rho_model(name); hz = RHO_CFG[name][2]; JDyn.obc_ho_regime(JDyn.Occbin(m); horizon=hz, check_ahead=hz, kw...))
mklcp(name; kw...) = (m = rho_model(name); hz = RHO_CFG[name][2]; JDyn.obc_ho(JDyn.Occbin(m); horizon=hz, check_ahead=hz, m_diagnostic=false, kw...))

