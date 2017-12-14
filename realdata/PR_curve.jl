@everywhere function PR_curve(users_to_test,rating_set::RatingSet,model::RatingsModel)

  K=101
  r=1
  u1=0.359
  Nb_users_test=length(users_to_test)
  Results=zeros(12,10)
  iter_th=0



  for th in linspace(2.4,4.2,10)
    iter_th+=1
    res_th=zeros(12,Nb_users_test)
    iter_u=0
    for user in users_to_test
      delta,wdelta,n,wn=learn_stats(user,K,rating_set,model)
      sb,wsb,naive,weighted=predict(n,wn,delta,wdelta,u1,sigma,r)
      iter_u+=1
      res_th[:,iter_u]=eval_PR(user,sb,wsb,naive,weighted,th,rating_set)
    end

    oks_p=vec(!isnan(res_th[1,:]) & !isnan(res_th[2,:]) & !isnan(res_th[3,:]) & !isnan(res_th[4,:]))
    Results[1:4,iter_th]=mean(res_th[1:4,oks_p],2)
    oks_r=vec(!isnan(res_th[5,:]) & !isnan(res_th[6,:]) & !isnan(res_th[7,:]) & !isnan(res_th[8,:]))
    Results[5:8,iter_th]=mean(res_th[5:8,oks_r],2)
    oks_f=vec(!isnan(res_th[9,:]) & !isnan(res_th[10,:]) & !isnan(res_th[11,:]) & !isnan(res_th[12,:]))
    Results[9:12,iter_th]=mean(res_th[9:12,oks_f],2)

    println("experiments finished for th=$th .")
  end

  return Results
end


@everywhere function Prec_at_tau(users_to_test,rating_set::RatingSet,model::RatingsModel)

  K=101
  r=1
  u1=0.27
  Nb_users_test=length(users_to_test)
  sigma=1
  

  P_allusers=@parallel ((x,y)->vcat(x,y)) for user in users_to_test
    delta,wdelta,n,wn=learn_stats(user,K,rating_set,model)
    sb,wsb,naive,weighted=predict(n,wn,delta,wdelta,u1,sigma,r)
    res=zeros(2,10)
    iter_th=1
    for th in linspace(3,4.5,10)
      res[:,iter_th]=prec_tau(user,sb,naive,th,rating_set)
      iter_th+=1
    end
    [res[1,:] res[2,:]]
  end

  return P_allusers
end

