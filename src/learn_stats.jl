# Build statistics over all users giving the delta and n vectors that will be used to predict the ratins with the selection bias assumption

function learn_stats(user,K,ratings_set::RatingSet, model::RatingModel)
  num_users=length(users(model))
  num_items=length(items(model))
  features = user_features(model, user)


  n=zeros(num_items)
  delta=zeros(num_items)
  wdelta=zeros(num_items)

  sim_us=IncrementalSVD.similar_users(model, user,max_results=K)
  for u in sim_us
    rats=user_ratings(ratings_set,u)
    similarity=cosine_similarity(user_features(model, u), features)
    for (item,rating) in rats
      n[item]+=1
      delta[item]+=rating
      wdelta[item]+=similarity*rating
    end
  end


  return Stats(delta,wdelta,n)
end
