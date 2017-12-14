
@everywhere function likelihood(theta::Vector,beta::Vector, n,delta,sigma )
    ll=sum(theta.^2.*n./(2*sigma^2)-theta.*delta/(sigma^2)-n.*beta)+sum(n)*log(sum(exp(beta)))
    return ll
end

@everywhere function objective(param::Vector,r, u1, likelihood)
    size=length(param)
    theta=param[1:size/2]
    beta=param[size/2+1:end]
    penal=r*sum((u1*beta-theta).^2)
    return likelihood(theta,beta)+penal
end

@everywhere function my_gradient!(param::Vector, storage::Vector,n,delta,sigma,r, u1)
    size=int(length(param))
    theta=param[1:size/2]
    beta=param[size/2+1:end]
    storage[1:size/2]=theta.*n/(sigma^2)-delta/(sigma^2)+2*r*(theta-u1*beta)
    storage[size/2+1:end]=-n+sum(n)*exp(beta)/sum(exp(beta))-2*r*u1*(theta-u1*beta)

end

@everywhere function local_objective(r, u1,likelihood)
  return param->objective(param,r,u1,likelihood)

end


@everywhere  function local_gradient(n,delta, sigma,r,u1)
  return (param, storage)->my_gradient!(param,storage,n,delta,sigma,r,u1)

end

@everywhere  function local_likelihood(n,delta, sigma)
  return (theta,beta)->likelihood(theta,beta,n,delta,sigma)

end

@everywhere function predict(n,wn,delta,wdelta,u1,sigma,r)

  ok_movs=find(n)
  sb=3.51*ones(length(n)) #zeros(length(n))
  wsb=3.51*ones(length(n)) #zeros(length(n))
  naive=3.51*ones(length(n)) #zeros(length(n))
  weighted=3.51*ones(length(n)) #zeros(length(n))

  # naive averages
  naive[ok_movs]=delta[ok_movs]./n[ok_movs]

  #weighted averages
  weighted[ok_movs]=wdelta[ok_movs]./n[ok_movs]

  # learning with selection bias
  init=zeros(2*length(ok_movs))
  init=vec([delta[ok_movs]'./n[ok_movs]' delta[ok_movs]'./(n[ok_movs]*u1)'])

  loc_l=local_likelihood(n[ok_movs],delta[ok_movs],sigma)
  loc_obj=local_objective(r,u1,loc_l)
  loc_grad=local_gradient(n[ok_movs],delta[ok_movs],sigma,r,u1)

  d=DifferentiableFunction(loc_obj,loc_grad)

  sol_temp=Optim.l_bfgs(d, init,grtol=1e-4)
  sb[ok_movs]=sol_temp.minimum[1:length(ok_movs)]

  #learning with weighted selection bias
  loc_l=local_likelihood(wn[ok_movs],wdelta[ok_movs],sigma)
  loc_obj=local_objective(r,u1,loc_l)
  loc_grad=local_gradient(wn[ok_movs],wdelta[ok_movs],sigma,r,u1)

  d=DifferentiableFunction(loc_obj,loc_grad)

  sol_temp=Optim.l_bfgs(d, init,grtol=1e-4)
  wsb[ok_movs]=sol_temp.minimum[1:length(ok_movs)]


  return sb,wsb,naive,weighted


end
