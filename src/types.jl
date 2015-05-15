
# An internal type used only by the train function for caching between epochs.
type Stats
    delta::Array{Float32,1}
    n::Array{Float32,1}
    wdelta::Array{Float32,1}
end

type Predicted
  sb::Array{Float32,1}
  naive::Array{Float32,1}
  weighted::Array{Float32,1}
end
