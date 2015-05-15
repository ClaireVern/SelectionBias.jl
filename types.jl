# A single rating. The user and item are both represented by integer ids. A map
# between ids and user/item names is stored elsewhere, both in the rating set
# and the model.
type Rating
    user::Int32
    item::Int32
    value::Float32
end

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
