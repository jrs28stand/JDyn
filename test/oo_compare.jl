# Compare a JDyn results struct with a struct read from an Octave .mat file (test helper; no side effects)
# Compare a JDyn MStruct against a struct read from an Octave .mat file: returns (rows of problems, max numeric diff)
function oo_compare(j, r; path="oo_", skip=Set{String}(), rtol=0.0, atol=0.0)
    probs = String[]; worst = Ref(0.0); worstat = Ref("")
    function rec(j, r, p)
        if r isa AbstractDict
            if !(j isa JDyn.MStruct)
                push!(probs, "$p: expected a struct, got $(typeof(j))"); return
            end
            for (k, rv) in r
                pk = p * "." * k
                any(s -> occursin(s, pk), skip) && continue
                if !haskey(j, k)
                    push!(probs, "$pk: missing")
                else
                    rec(j[k], rv, pk)
                end
            end
            for k in keys(j)
                haskey(r, k) || any(s -> occursin(s, p * "." * k), skip) || push!(probs, "$(p).$k: extra field in JDyn")
            end
        elseif r isa AbstractString
            j == r || push!(probs, "$p: $(repr(j)) vs $(repr(r))")
        elseif r isa Bool
            (j isa Bool && j == r) || push!(probs, "$p: $(repr(j)) vs $(repr(r))")
        elseif r isa Number || r isa AbstractArray
            if !(j isa Number || j isa AbstractArray)
                push!(probs, "$p: expected numeric, got $(typeof(j))"); return
            end
            ja = j isa Number ? fill(j, 1, 1) : j
            ra = r isa Number ? fill(r, 1, 1) : r
            if size(ja) != size(ra)
                # MAT.jl drops trailing singleton dims / 0x0
                push!(probs, "$p: size $(size(ja)) vs $(size(ra))"); return
            end
            isempty(ra) && return
            if endswith(p, ".eigval")
                a = sort(vec(ComplexF64.(ja)); by=z -> (round(real(z); digits=9), round(imag(z); digits=9)))
                b = sort(vec(ComplexF64.(ra)); by=z -> (round(real(z); digits=9), round(imag(z); digits=9)))
                d = maximum(abs.(a .- b))
            else
                both_nan = isnan.(ja) .& isnan.(ra)
                a = ifelse.(both_nan, 0.0, ComplexF64.(ja)); b = ifelse.(both_nan, 0.0, ComplexF64.(ra))
                d = maximum(abs.(a .- b))
                any(isnan.(ja) .!= isnan.(ra)) && (d = Inf)
            end
            if d > worst[]; worst[] = d; worstat[] = p; end
            d > atol + rtol * maximum(abs.(ra)) && push!(probs, "$p: max diff $d")
        else
            push!(probs, "$p: unsupported reference type $(typeof(r))")
        end
    end
    rec(j, r, path)
    probs, worst[], worstat[]
end
