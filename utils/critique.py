from gurobipy import *
from prediction.predictor import predict_scores
from utils.progress import inhour

import time
import numpy as np


def sample_users(num_users, num_users_sampled):
    return np.random.choice(num_users, num_users_sampled, replace=False)

def sample_items(candidate_items, num_items_sampled):
    return np.random.choice(candidate_items, size=num_items_sampled, replace=False)

def sample_keyphrase():
    # Critique the most popular keyphrases in the remaining
    critiqued_keyphrase = remaining_keyphrases[np.argmax(self.keyphrase_popularity[remaining_keyphrases])]

def get_max_length(df, num_keyphrases, max_iteration):
    df_s_f = df[(df['result'] == 'successful') | (df['result'] == 'fail')]
    df_s_f.loc[df_s_f['num_existing_keyphrases'] > max_iteration, 'num_existing_keyphrases'] = max_iteration
    return df_s_f['num_existing_keyphrases'].mean()

def get_average_length(df, n):
    df_s_f = df[(df['result'] == 'successful') | (df['result'] == 'fail')]
    iteration = df_s_f[df_s_f['target_rank']==n].groupby('user_id', as_index=False).agg({'iteration':'mean'})['iteration'].to_numpy()
    return (np.average(iteration), 1.96*np.std(iteration)/np.sqrt(len(iteration)))

def get_success_num(df, n):
    return len(df[(df['result'] == 'successful') & (df['target_rank'] == n)])

def get_fail_num(df, n):
    return len(df[(df['result'] == 'fail') & (df['target_rank'] == n)])

def get_success_rate(df, n):
    df_s_f = df[(df['result'] == 'successful') | (df['result'] == 'fail')]
    df_list_result = df_s_f[df_s_f['target_rank']==n].groupby('user_id', as_index=False)['result'].apply(list).reset_index(name='result')
    successful_rate = df_list_result['result'].apply(lambda r: r.count("successful")/len(r)).to_numpy()
    return (np.average(successful_rate), 1.96*np.std(successful_rate)/np.sqrt(len(successful_rate)))

def count_occurrence(x):
    return x.count("successful")

def add_pop(x, item_pop_index):
    return np.where(item_pop_index == x)[0][0]


##################################
# Baseline Methods 
##################################
def LPUAC(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, query, test_user, item_latent, reg):

    critiqued_vector = np.zeros(keyphrase_freq[0].shape)

    for q in query:
        critiqued_vector[q] = -keyphrase_freq[test_user][q]

    num_critiques = len(query)

    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

    # UAC
    lambdas = []

    for k in range(num_critiques):
        optimal_lambda = 1/(1+num_critiques) # All equals to 1/(K+1)
        lambdas.append(optimal_lambda)
        critiqued_vector[query[k]] *= optimal_lambda

    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)

    new_prediction = (1/(1+num_critiques))*initial_prediction_u + critique_score.flatten()

    return new_prediction, lambdas

def LPBAC(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, query, test_user, item_latent, reg):

    critiqued_vector = np.zeros(keyphrase_freq[0].shape)

    for q in query:
        critiqued_vector[q] = -keyphrase_freq[test_user][q]

    num_critiques = len(query)

    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

    # UAC
    lambdas = []

    for k in range(num_critiques):
        optimal_lambda = 1/(2*num_critiques) # All equals to 1/(2*K)
        lambdas.append(optimal_lambda)
        critiqued_vector[query[k]] *= optimal_lambda

    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)

    new_prediction = (1/2)*initial_prediction_u + critique_score.flatten()

    return new_prediction, lambdas
##################################
# LP Objectives 
##################################

def LP1SimplifiedOptimize(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, query, test_user, item_latent, reg):

    critiqued_vector = np.zeros(keyphrase_freq[0].shape)

    for q in query:
        critiqued_vector[q] = -max(keyphrase_freq[test_user][q],keyphrase_freq[test_user].mean())

    num_critiques = len(query)

    W2 = reg.coef_
    W = item_latent.dot(W2)

    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

    start_time = time.time()

    # Model
    m = Model("LP1Simplified")
    m.setParam('OutputFlag', 0) 

    # Assignment variables
    lambs = []

    for k in range(num_critiques):
        lambs.append(m.addVar(lb=-1,
                              ub=1,
                              vtype=GRB.CONTINUOUS,
                              name="lamb%d" % query[k]))

    m.setObjective(quicksum(initial_prediction_u[affected_item] * num_unaffected_items + quicksum(lambs[k] * critiqued_vector[query[k]] * W[affected_item][query[k]] * num_unaffected_items for k in range(num_critiques)) for affected_item in affected_items) - quicksum(initial_prediction_u[unaffected_item] * num_affected_items + quicksum(lambs[k] * critiqued_vector[query[k]] * W[unaffected_item][query[k]] * num_affected_items for k in range(num_critiques)) for unaffected_item in unaffected_items), GRB.MINIMIZE)

    # Optimize
    m.optimize()

    # print("Elapsed: {}".format(inhour(time.time() - start_time)))

    lambdas = []
    for k in range(num_critiques):
        optimal_lambda = m.getVars()[k].X
        lambdas.append(optimal_lambda)
        critiqued_vector[query[k]] *= optimal_lambda

    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)

    new_prediction = initial_prediction_u + critique_score.flatten()

    return new_prediction, lambdas



##################################
# LP-Ranking
##################################

def LPRank(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, 
            query, test_user, item_latent, reg, Z_pre, Z, thetas_pre, lamb = 10, bound_range = 2):
    """
    Incremental Approach 
    """
    critiqued_vector = np.zeros(keyphrase_freq[0].shape)
    for q in query:
        critiqued_vector[q] = -max(keyphrase_freq[test_user][q],keyphrase_freq[test_user].mean())

    num_critiques = len(query)
    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

    # Model
    m = Model("LPRank")
    m.setParam('OutputFlag', 0) # set to 1 for outputing details
    
    # Assignment variables
    thetas = []
    us = []
    xi_pos = []
    xi_neg = []
    # weight thetas
    for k in range(num_critiques + 1):
        thetas.append(m.addVar(lb=-bound_range,
                              ub=bound_range,
                              vtype=GRB.CONTINUOUS,
                              name="theta%d" % k))
    thetas = np.array(thetas)
    
    # dummy variable u for absolute theta
    for k in range(num_critiques + 1):
        us.append(m.addVar(vtype=GRB.CONTINUOUS,
                          name="u%d" % k))
    
    # slack variables xi
    for i in range(num_affected_items):
        xi_pos.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_pos%d" % i ))
    for i in range(num_unaffected_items):
        xi_neg.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_neg%d" % i ))

    ## constraints
    # constraints for dummy variable u's

    for k in range(num_critiques+1):
        m.addConstr(us[k] >= thetas[k] - 1/(num_critiques+1))
        m.addConstr(us[k] >= 1/(num_critiques+1) - thetas[k])

    # Affected items rank higher
    for j in range(num_affected_items):
        m.addConstr( thetas_pre.dot(Z_pre.dot(item_latent[affected_items[j]])) - thetas.dot(Z.dot(item_latent[affected_items[j]])) >=  1 - xi_pos[j], name = "pos_constraint%d" % j )

    for j in range(num_unaffected_items):
        m.addConstr( thetas.dot(Z.dot(item_latent[unaffected_items[j]]))  >=  thetas_pre.dot(Z_pre.dot(item_latent[unaffected_items[j]])) + 1 - xi_neg[j], name = "neg_constraint%d" % j )



        
    m.setObjective(quicksum(us) + lamb * (quicksum(xi_pos)+quicksum(xi_neg)), GRB.MINIMIZE)  # Single regularization

    # Optimize
    m.optimize()

    # Save optimal thetas
    thetas = []
    for k in range(num_critiques+1):
        optimal_theta = m.getVarByName("theta%d" % k).X
        thetas.append(optimal_theta)
    
    for k in range(num_critiques):
        critiqued_vector[query[k]] *= thetas[k+1]

    # Get rating score
    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)
    new_prediction = thetas[0]*initial_prediction_u + critique_score.flatten()
    
    return new_prediction, thetas


def LPRank2(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, 
            query, test_user, item_latent, reg, Z_pre, Z, lamb = 10, bound_range = 2):
    """
    Non Incremental Approach 
    """
    critiqued_vector = np.zeros(keyphrase_freq[0].shape)
    for q in query:
        critiqued_vector[q] = -max(keyphrase_freq[test_user][q],keyphrase_freq[test_user].mean())

    num_critiques = len(query)
    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

    # Model
    m = Model("LPRank")
    m.setParam('OutputFlag', 0) # set to 1 for outputing details
    
    # Assignment variables
    thetas = []
    us = []
    xi_pos = []
    xi_neg = []
    # weight thetas
    for k in range(num_critiques + 1):
        thetas.append(m.addVar(lb=-bound_range,
                              ub=bound_range,
                              vtype=GRB.CONTINUOUS,
                              name="theta%d" % k))
    thetas = np.array(thetas)
    
    # dummy variable u for absolute theta
    for k in range(num_critiques + 1):
        us.append(m.addVar(vtype=GRB.CONTINUOUS,
                          name="u%d" % k))
    
    # slack variables xi
    for i in range(num_affected_items):
        xi_pos.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_pos%d" % i ))
    for i in range(num_unaffected_items):
        xi_neg.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_neg%d" % i ))

    ## constraints
    # constraints for dummy variable u's

    for k in range(num_critiques+1):
        m.addConstr(us[k] >= thetas[k] - 1/(num_critiques+1))
        m.addConstr(us[k] >= 1/(num_critiques+1) - thetas[k])

    # Affected items rank higher
    for j in range(num_affected_items):
        m.addConstr( initial_prediction_u[affected_items[j]] - thetas.dot(Z.dot(item_latent[affected_items[j]])) >=   1 - xi_pos[j], name = "pos_constraint%d" % j )
    
    # Unaffected items rank lower
    for j in range(num_unaffected_items):
        m.addConstr( thetas.dot(Z.dot(item_latent[unaffected_items[j]]))  >=  initial_prediction_u[unaffected_items[j]] + 1 - xi_neg[j], name = "neg_constraint%d" % j )

        
    m.setObjective(quicksum(us) + lamb * (quicksum(xi_pos)+quicksum(xi_neg)), GRB.MINIMIZE)  # Single regularization

    # Optimize
    m.optimize()

    # Save optimal thetas
    thetas = []
    for k in range(num_critiques+1):
        optimal_theta = m.getVarByName("theta%d" % k).X
        thetas.append(optimal_theta)
    
    for k in range(num_critiques):
        critiqued_vector[query[k]] *= thetas[k+1]

    # Get rating score
    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)
    new_prediction = thetas[0]*initial_prediction_u + critique_score.flatten()
    
    return new_prediction, thetas

