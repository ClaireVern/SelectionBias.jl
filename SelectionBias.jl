module SelectionBias

#using ProgressMeter
using IncrementalSVD

#import IncrementalSVD


export learn_stats
export predict
export Stats, Predicted

#export split_ratings, rmse, cosine_similarity
#export items, users, item_features, user_features, show_items_by_features, item_search
#export similar_items, similar_users, user_ratings, get_predicted_rating
#export  load_small_movielens_dataset, load_large_movielens_dataset
#export Rating, RatingSet


include("types.jl")
include("learn_stats.jl")
#include("util.jl")
include("sb_source.jl")

#include("book_crossing_dataset.jl")
#include("movielens_dataset.jl")

end
