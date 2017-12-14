@everywhere using Optim
println("load IncrementalSVD package")
@everywhere using IncrementalSVD
@everywhere using ProgressMeter
@everywhere using Distributions

println("load my functions")
include("learn_stats.jl")
include("sb_source.jl")
include("eval.jl")
include("tune_r.jl")
include("PR_curve.jl")
K=101

#println("build rating_set : load and split data")
#rating_set = IncrementalSVD.load_small_movielens_dataset(0.25);

#uncomment when working on ML10M
rating_set = IncrementalSVD.load_large_movielens_dataset(0.25);

println("compute SVD model that will be used for computing neighborhoods and similarity")
model = IncrementalSVD.train(rating_set, 25);

all_users=[k for k in IncrementalSVD.users(model)]
#u1=0.35
#uncomment when working on ML10M
u1=0.27
sigma=1.5
#error_r=tune_r(rating_set,model,100,4)
r=1
N=15
Nb_users=10000

users_to_test=sample(1:length(all_users),Nb_users,replace=false)
users_to_test=all_users[users_to_test]

segs=100
batch=int(Nb_users/segs)
#println("doing experiments on $Nb_users users")
p=Progress(segs,1)
for s in 1:segs


  Results=@parallel ((x,y)-> hcat(x,y)) for user in users_to_test[((s-1)*batch+1):s*batch]


    #println(user)
    delta,wdelta,n,wn=learn_stats(user,K,rating_set,model)
    sb,wsb,naive,weighted=predict(n,wn,delta,wdelta,u1,sigma,r)
    Res_rmse=rmse_all(user,sb,wsb,naive,weighted,rating_set)
    Res_PR=eval_PR(user,sb,wsb,naive,weighted,4,rating_set)
    Prec_at_N=eval_at_N(user,sb,wsb,naive,weighted,N,rating_set,model)
    [Res_rmse;Res_PR;Prec_at_N]

  end

  #writedlm("Results_ML10M__K$K _r$r _Nb_us$Nb_users _seg$s.csv",Results)
  writedlm("Results_ML10M__K$K _r$r _Nb_us$Nb_users _seg$s.csv",Results)
  next!(p)
end


println("done and saved.")
