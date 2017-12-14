# Build statistics over all users giving the delta and n vectors that will be used to predict the ratins with the selection bias assumption

using IncrementalSVD


@everywhere function learn_stats(user,K,rating_set::RatingSet, model::RatingsModel)
  num_users=length(users(model))
  num_items=length(items(model))
  features = user_features(model, user)


  n=zeros(num_items)
  wn=zeros(num_items)
  delta=zeros(num_items)
  wdelta=zeros(num_items)

  sim_us=IncrementalSVD.similar_users(model, user,max_results=K)
  for u in sim_us[2:end]
    rats=user_ratings(rating_set,u)
    similarity=cosine_similarity(user_features(model, u), features)
    for (item,rating) in rats
      n[rating_set.item_to_index[item]]+=1
      wn[rating_set.item_to_index[item]]+=similarity
      wdelta[rating_set.item_to_index[item]]+=similarity*rating
      delta[rating_set.item_to_index[item]]+=rating

    end
  end


  return (delta,wdelta,n,wn)
end
