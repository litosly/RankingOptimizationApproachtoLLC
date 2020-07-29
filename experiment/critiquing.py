from utils.critique import sample_users
from utils.modelnames import critiquing_models

import numpy as np
import pandas as pd


def critiquing(matrix_Train, matrix_Test, keyphrase_freq, dataset_name, model,
               parameters_row, critiquing_model_name, item_keyphrase_freq=None, num_users_sampled=10,
               num_items_sampled=5, max_iteration_threshold=10,topk=10,lamb=10,keyphrase_selection_method="pop"):

    num_users = matrix_Train.shape[0]
    num_keyphrases = keyphrase_freq.shape[1]

    keyphrase_popularity = np.sum(item_keyphrase_freq, axis=1)

    columns = ['user_id', 'item_id', 'target_rank', 'iteration', 'critiqued_keyphrase', 'item_rank', 'item_score', 'num_existing_keyphrases', 'result', 'lambda']
    df = pd.DataFrame(columns=columns)

    row = {}

    target_ranks = [1]

    # Randomly select test users
    np.random.seed(1201)
    test_users = np.random.choice(num_users, num_users_sampled, replace=False)
    
    critiquing_model = critiquing_models[critiquing_model_name](keyphrase_freq=keyphrase_freq,
                                                                item_keyphrase_freq=item_keyphrase_freq,
                                                                row=row,
                                                                matrix_Train=matrix_Train,
                                                                matrix_Test=matrix_Test,
                                                                test_users=test_users,
                                                                target_ranks=target_ranks,
                                                                num_items_sampled=num_items_sampled,
                                                                num_keyphrases=num_keyphrases,
                                                                df=df,
                                                                max_iteration_threshold=max_iteration_threshold,
                                                                keyphrase_popularity=keyphrase_popularity,
                                                                dataset_name=dataset_name,
                                                                model=model,
                                                                parameters_row=parameters_row,
                                                                topk=topk,
                                                                lamb=lamb,
                                                                keyphrase_selection_method=keyphrase_selection_method)

    df = critiquing_model.start_critiquing()

    return df

