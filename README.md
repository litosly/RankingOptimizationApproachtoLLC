# RankingOptimizationApprochtoLLC
Official Code Framework of the paper "A Ranking Optimization Approach to Latent Linear Critiquing in Conversational Recommender Systems"

If you are interested in building up your research on this work, please cite:

```
@inproceedings{Li:recsys20,
  author    = {Hanze Li and Scott Sanner and Kai Luo and Ga Wu},
  title     = {A Ranking Optimization Approach to Latent Linear Critiquing in Conversational Recommender System},
  booktitle = {Proceedings of the 14th International {ACM} Conference on Recommender Systems {(RecSys-20)}},
  address   = {Online},
  year      = {2020}, 
  url_paper = {https://doi.org/10.1145/3383313.3412240}
}
```

# Author Affiliate
<p align="center">
<a href="https://www.utoronto.ca//"><img src="https://github.com/k9luo/DeepCritiquingForVAEBasedRecSys/blob/master/logos/U-of-T-logo.svg" height="80"></a> | 
<a href="https://vectorinstitute.ai/"><img src="https://github.com/k9luo/DeepCritiquingForVAEBasedRecSys/blob/master/logos/vectorlogo.svg" height="80"></a> | 
</p>

# Algorithm Implemented
1. Ranking-based LLC

# Dataset
1. Yelp,
2. Beer Advocate,

We don't have rights to release the datasets. Please ask permission from Professor [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/).

Please refer to the [`preprocess` folder](https://github.com/wuga214/DeepCritiquingForRecSys/tree/master/preprocess) for preprocessing raw datasets steps.
Please refer to https://github.com/k9luo/LatentLinearCritiquingforConvRecSys for previous codebase for Latent Liear Critiquing.
This repo adapts PLRec implementation from NCE-PLRec repo. The full version of the recommender system could be found in [https://github.com/wuga214/NCE_Projected_LRec]

# Keyphrase
Keyphrases we used are not necessarily the best. If you are interested in how we extracted those keyphrases, please refer to the [`preprocess` folder](https://github.com/wuga214/DeepCritiquingForRecSys/tree/master/preprocess). If you are interested in what keyphrases we extracted, please refer to the [`data` folder](https://github.com/litosly/RankingOptimizationApproachtoLLC/tree/master/data).

# Packages
We use Gurobi Optimizer to train our LP Objectives, please see usage in [website](https://www.gurobi.com/).

# Example Commands
Note: change --data_dir and --dataset_name for different dataset, change --topk for different number of top-K (K = 100 by default) rated items during optimization, change --lamb for different regularization parameter in equation(12). 
### Reproduce Critiquing
```
python critiquing_experiment.py --data_dir "data/yelp/" --dataset_name "yelp" --save_path "test_result/beer/llc_rank_topk10_lamb10_diff.csv" --topk 10 --lamb 10 --keyphrase_selection_method diff
```
### Hyperparameter Tuning 
```
python tuning_experiment.py --data_dir "data/beer/" --dataset_name "beer" --save_path "tuning_lambda/beer/" 
```
