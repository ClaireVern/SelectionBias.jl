function tune_r(rating_set::RatingSet,model::RatingsModel,Nb_users_test,max_value)
  test_users=[k for k in IncrementalSVD.users(model)]
  test_users=test_users[1:Nb_users_test]
  error_r=vec(zeros(1,20))
  iter_r=0
  for r in linspace(0.1,max_value,10)
    iter_r+=1
    for user in test_users
      delta,wdelta,n=learn_stats(user,K,rating_set,model)
      sb,naive,weighted=predict(n,delta,wdelta,u1,sigma,r)
      error_r[iter_r]+=rmse_all(user,sb,naive,weighted,rating_set)[2]
    end
    error_r[iter_r]=error_r[iter_r]/Nb_users_test
  end
  return error_r
end
