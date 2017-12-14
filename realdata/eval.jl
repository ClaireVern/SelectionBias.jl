using IncrementalSVD

@everywhere function test_set_user(user,rating_set::RatingSet)
  user_id = rating_set.user_to_index[user]
  extract_rating = rating -> (rating.item, rating.value)
  is_user = rating -> rating.user == user_id
  to_test=map(extract_rating, filter(is_user, rating_set.test_set))

  return to_test
end

@everywhere function train_set_user(user,rating_set::RatingSet)
  user_id = rating_set.user_to_index[user]
  extract_rating = rating -> (rating.item, rating.value)
  is_user = rating -> rating.user == user_id
  train=map(extract_rating, filter(is_user, rating_set.training_set))

  return train
end

@everywhere function rmse_all(user,sb,wsb,naive,weighted,rating_set::RatingSet)

  to_test=test_set_user(user,rating_set::RatingSet)
  len_test=length(to_test)

  rmse_sb=0.0
  rmse_wsb=0.0
  rmse_naive=0.0
  rmse_weighted=0.0

  for (item,rating) in to_test
    rmse_sb+=(sb[item]-rating)^2
    rmse_wsb+=(wsb[item]-rating)^2
    rmse_naive+=(naive[item]-rating)^2
    rmse_weighted+=(weighted[item]-rating)^2
  end
  rmse_sb=sqrt(rmse_sb/len_test)
  rmse_wsb=sqrt(rmse_wsb/len_test)
  rmse_naive=sqrt(rmse_naive/len_test)
  rmse_weighted=sqrt(rmse_weighted/len_test)

  return [user,rmse_sb,rmse_wsb,rmse_naive,rmse_weighted]

end

@everywhere function eval_PR(user,sb,wsb,naive,weighted,th,rating_set::RatingSet)
  user_id = rating_set.user_to_index[user]
  to_test=test_set_user(user,rating_set::RatingSet)

  TP=zeros(4)
  FP=zeros(4)
  FN=zeros(4)

  for (item,rating) in to_test
    if rating>3.7
      #this item is relevant and must be recommended
      sb[item]>=th ? TP[1]+=1 : FN[1]+=1
      wsb[item]>=th ? TP[2]+=1 : FN[2]+=1
      naive[item]>=th ? TP[3]+=1 : FN[3]+=1
      weighted[item]>=th ? TP[4]+=1 : FN[4]+=1
    else
      sb[item]>=th ? FP[1]+=1 : nothing
      wsb[item]>=th ? FP[2]+=1 : nothing
      naive[item]>=th ? FP[3]+=1 : nothing
      weighted[item]>=th ? FP[4]+=1 : nothing
    end
  end
    Prec=TP./(TP+FP)
    Rec=TP./(FN+TP)
    F=2.*Prec.*Rec./(Prec+Rec)

    return [Prec;Rec;F]

end

@everywhere function prec_tau(user,sb,naive,th,rating_set)
  user_id = rating_set.user_to_index[user]
  to_test=test_set_user(user,rating_set::RatingSet)

  TP=zeros(2)

  rec_SB=find(sb.>th)
  rec_LS=find(naive.>th)


  for (item,rating) in to_test
    if rating>3.7
      #this item is relevant and must be recommended
      sb[item]>=th ? TP[1]+=1 : nothing

      naive[item]>=th ? TP[2]+=1 : nothing
    end


  end
    Prec_SB=TP[1]./length(rec_SB)
    Prec_LS=TP[2]./length(rec_LS)
    [Prec_SB;Prec_LS]
end

@everywhere function eval_at_N(user,sb,wsb,naive,weighted,N,rating_set::RatingSet,model::RatingsModel)
  user_id = rating_set.user_to_index[user]
  to_test=test_set_user(user,rating_set)
  train_set=train_set_user(user,rating_set)
  #remove items that are in the train and that must not be recommended
  train=IntSet([train_set[i][1] for i in 1:length(train_set)])
  items_to_rec=IntSet([i for i in range(1,length(items(model)))])
  symdiff!(items_to_rec,train)

  sb_rec=sort([i for i in items_to_rec],by=(m-> -sb[m]))[1:N]
  wsb_rec=sort([i for i in items_to_rec],by=(m-> -wsb[m]))[1:N]
  naive_rec=sort([i for i in items_to_rec],by=(m-> -naive[m]))[1:N]
  weighted_rec=sort([i for i in items_to_rec],by=(m-> -weighted[m]))[1:N]
  TP=zeros(4)
  bin_prec=zeros(4)



  for (item,rating) in to_test
    if rating>=3.7

      #this item is relevant and must be recommended
      item in sb_rec ? TP[1]+=1 : nothing
      item in wsb_rec ? TP[2]+=1 : nothing
      item in naive_rec ? TP[3]+=1 : nothing
      item in weighted_rec ? TP[4]+=1 : nothing
    end
  end

  Prec_at_N=TP./N

  TP[1]>0 ? bin_prec[1]=1 : nothing
  TP[2]>0 ? bin_prec[2]=1 : nothing
  TP[3]>0 ? bin_prec[3]=1 : nothing
  TP[4]>0 ? bin_prec[4]=1 : nothing

  return [Prec_at_N;bin_prec]

end

@everywhere function baseline(user,n,N,rating_set::RatingSet,model::RatingsModel)
  user_id = rating_set.user_to_index[user]
  to_test=test_set_user(user,rating_set::RatingSet)
  train_set=train_set_user(user,rating_set)
  #remove items that are in the train and that must not be recommended
  train=IntSet([train_set[i][1] for i in 1:length(train_set)])
  items_to_rec=IntSet([i for i in range(1,length(items(model)))])
  symdiff!(items_to_rec,train)
  baseline_rec=sort([i for i in items_to_rec],by=(m-> -n[m]))[1:N]

  TP=0
  bin_prec=0
  for (item,rating) in to_test
    if rating>=3.7

      #this item is relevant and must be recommended
      item in baseline_rec ? TP+=1 : nothing
    end
  end

  Prec_at_N=TP./N

  TP[1]>0 ? bin_prec=1 : nothing
  return [Prec_at_N;bin_prec]

end


