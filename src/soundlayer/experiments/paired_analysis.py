"""Small-sample deterministic paired statistics using the standard library."""
import itertools,math,random,statistics
def bootstrap_ci(values,resamples=10000,seed=0):
    if not values:return None
    rng=random.Random(seed);n=len(values)
    means=sorted(sum(rng.choice(values) for _ in range(n))/n for _ in range(resamples))
    return [means[int(.025*(resamples-1))],means[int(.975*(resamples-1))]]
def sign_permutation_p(values):
    nz=[x for x in values if x]
    if not nz:return 1.0
    observed=abs(sum(nz)/len(nz));extreme=0;total=2**len(nz)
    for signs in itertools.product((-1,1),repeat=len(nz)):
        if abs(sum(x*s for x,s in zip(nz,signs))/len(nz))>=observed-1e-12:extreme+=1
    return extreme/total
def signed_rank_p(values):
    rounded=[round(x,8) for x in values if round(x,8)!=0]
    if not rounded:return 1.0
    ordered=sorted(enumerate(rounded),key=lambda x:abs(x[1]));ranks=[0]*len(rounded)
    for rank,(i,_) in enumerate(ordered,1):ranks[i]=rank
    observed=abs(sum(r if x>0 else -r for x,r in zip(rounded,ranks)))
    extreme=0
    for signs in itertools.product((-1,1),repeat=len(rounded)):
        if abs(sum(s*r for s,r in zip(signs,ranks)))>=observed:extreme+=1
    return extreme/(2**len(rounded))
def summarize(values,resamples=10000,seed=0):
    if not values:return {"n_available":0,"available":False,"unavailable_reason":"NO_PAIRED_VALUES"}
    return {"n_available":len(values),"available":True,"mean_delta":statistics.mean(values),
      "median_delta":statistics.median(values),"bootstrap_95_ci":bootstrap_ci(values,resamples,seed),
      "positive_case_count":sum(x>0 for x in values),"negative_case_count":sum(x<0 for x in values),
      "zero_case_count":sum(x==0 for x in values),"permutation_p":sign_permutation_p(values),
      "wilcoxon_p":signed_rank_p(values),"effect_direction":"POSITIVE" if statistics.mean(values)>0 else "NEGATIVE" if statistics.mean(values)<0 else "ZERO",
      "availability_denominator":len(values)}
def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);out=[0]*len(pvalues);running=0
    for rank,i in enumerate(order):
        running=max(running,min(1.0,(len(pvalues)-rank)*pvalues[i]));out[i]=running
    return out
