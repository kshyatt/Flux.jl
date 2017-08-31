struct SGD
  ps::Vector{Param}
  η::Float32
end

sgd(m, η) = SGD(params(m), η)

function update!(o::SGD)
  for p in o.ps
    x, Δ = p.x, p.Δ
    x .-= Δ .* o.η
    Δ .= 0
  end
end
