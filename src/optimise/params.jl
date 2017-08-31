using DataFlow: OSet

children(x) = ()

params(ps, m) = foreach(m -> params(ps, m), children(m))

function params(m)
  ps = OSet()
  params(ps, m)
  return collect(ps)
end

struct Param{T}
  x::T
  Δ::T
end

Param(x) = Param(x, zeros(x))
