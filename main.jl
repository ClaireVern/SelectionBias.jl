import IncrementalSVD

import SelectionBias


K=50

rating_set = IncrementalSVD.load_small_movielens_dataset();

model = IncrementalSVD.train(rating_set, 25);
user="1"

stats=learn_stats(user,K,ratings_set,model)

predicted=predict(stats)
