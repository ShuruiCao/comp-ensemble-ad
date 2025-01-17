from scipy import stats
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import weightedtau

def topKprec(scores,labels,topk):
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:topk]
    true_positives = np.sum(labels[top_indices])
    precision = true_positives / topk
    return precision
# LOF STYLE LOSS
from sklearn.neighbors import NearestNeighbors

# Function to calculate final loss for each item in the embedding_dict
def compute_final_loss(embedding_dict, loss_dict, k):
    # Get the keys and embeddings
    keys = list(embedding_dict.keys())
    embeddings = np.array([embedding_dict[key] for key in keys])

    # Reshape embeddings if necessary
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(len(keys), -1)

    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Compute avg_diff_j for each item and its neighbors
    avg_diffs = []
    for i in range(len(keys)):
        item_loss = loss_dict[keys[i]]
        neighbor_indices = indices[i][1:]
        neighbor_losses = [loss_dict[keys[idx]] for idx in neighbor_indices]
        avg_diff = np.mean([abs(item_loss - loss) for loss in neighbor_losses])
        avg_diffs.append(avg_diff)

    # Calculate final losses for each item
    final_losses = {}
    for i in range(len(keys)):
        avg_diff_i = avg_diffs[i]
        avg_diff_j = np.mean([avg_diffs[j] for j in indices[i][1:]])

        # # Calculate the final loss with a check for division by zero
        # if avg_diff_j != 0:
        #     final_loss = avg_diff_i / avg_diff_j
        # else:
        #     final_loss = avg_diff_i / 1
        final_loss = avg_diff_i / avg_diff_j
        final_losses[keys[i]] = final_loss

    return final_losses
# AOM on scores (numbers)
# select q (2) from Q (6), we have 15 buckets, take the average of max of these buckets
import itertools
import random

# Function to calculate the AOM using random buckets
def calculate_random_aom(score_lists, q):

    # Calculate all possible unique combinations of score lists
    combinationsS = list(itertools.combinations(score_lists, q))
    
    # Shuffle the combinations to randomize the order
    random.shuffle(combinationsS)
    
    # Calculate the max values for each combination
    max_values = [[max(items) for items in zip(*combination)] for combination in combinationsS]
    
    # Calculate the average of the max values
    final_scores = [sum(items) / len(combinationsS) for items in zip(*max_values)]
    return final_scores

from scipy import stats

# Function to calculate the AOM using random buckets on inverse ranks
def calculate_random_aom_rank(score_lists, q, limit=1000):
    # Convert scores to inverse ranks
    inverse_rank_lists = [1 / stats.rankdata(-score_list) for score_list in score_lists]
    
    # Calculate all possible unique combinations of inverse rank lists
    all_combinations = list(itertools.combinations(inverse_rank_lists, q))
    
    # Randomly select a subset of combinations
    combinationsS = random.sample(all_combinations, min(limit, len(all_combinations)))
    
    # Calculate the max values for each combination
    max_values = [[max(items) for items in zip(*combination1)] for combination1 in combinationsS]
    
    # Calculate the average of the max values
    final_scores = [sum(items) / len(combinationsS) for items in zip(*max_values)]
    return final_scores

# Function to calculate the AOM using random buckets on inverse ranks
def calculate_random_aomean_rank(score_lists, q, limit=1000):
    # Convert scores to inverse ranks
    inverse_rank_lists = [1 / stats.rankdata(-score_list) for score_list in score_lists]
    
    # Calculate all possible unique combinations of inverse rank lists
    all_combinations = list(itertools.combinations(inverse_rank_lists, q))
    
    # Randomly select a subset of combinations
    combinationsS = random.sample(all_combinations, min(limit, len(all_combinations)))
    
    # Calculate the max values for each combination
    mean_values = [[np.mean(items) for items in zip(*combination1)] for combination1 in combinationsS]
    
    # Calculate the average of the max values
    final_scores = [sum(items) / len(combinationsS) for items in zip(*mean_values)]
    return final_scores

def get_normalized_ap_diff(ap_diff):
    max_ap_diff = np.max(np.abs(ap_diff))
    return ap_diff/(max_ap_diff+0.00000001)

# Function to compute pairwise differences
def compute_pairwise_differences(scores):
    # Convert the list to a numpy array
    scores = np.array(scores)

    # Create a matrix with the pairwise differences
    differences_matrix = scores[:, None] - scores[None, :]

    # Extract upper triangular elements to get unique pairwise differences
    differences = differences_matrix[np.triu_indices(len(scores), k=1)]
    
    return differences

def weighted_kendall_from_pairs(a, b):
    c1_ind = np.abs(a)<=np.abs(b)
    c2_ind = np.abs(a)>np.abs(b)
    c1 = a/(b+0.0000001)
    c2 = b/(a+0.0000001) 
    c = np.zeros([len(a),])
    c[c1_ind] = c1[c1_ind]
    c[c2_ind] = c2[c2_ind]
    
    return np.sum(c)/np.sum(np.abs(c))
import matplotlib.pyplot as plt



def jaccard_similarity_IR(list1, list2, top_k=500):
    # Convert scores to inverse ranks
    rank1 = 1 / stats.rankdata(-np.array(list1))
    rank2 = 1 / stats.rankdata(-np.array(list2))

    # Select the top 500 samples in each ranking
    top_rank1 = set(np.argsort(rank1)[-top_k:])
    top_rank2 = set(np.argsort(rank2)[-top_k:])

    # Calculate Jaccard similarity
    return len(top_rank1.intersection(top_rank2)) / len(top_rank1.union(top_rank2))

def jaccard_similarity(list1, list2, top_k=500):
    # Rank the scores
    rank1 = stats.rankdata(list1)
    rank2 = stats.rankdata(list2)

    # Select the top 500 samples in each ranking
    top_rank1 = set(np.argsort(-rank1)[:top_k])
    top_rank2 = set(np.argsort(-rank2)[:top_k])

    # Calculate Jaccard similarity
    return len(top_rank1.intersection(top_rank2)) / len(top_rank1.union(top_rank2))
def ruzicka_similarity(list1, list2, top_k=500):
    # Convert lists to numpy arrays for easier manipulation
    array1 = np.array(list1)
    array2 = np.array(list2)

    # Get the indices of the top k scores
    top_indices1 = np.argsort(-array1)[:top_k]
    top_indices2 = np.argsort(-array2)[:top_k]

    # Get the union of indices from both top k lists
    union_indices = np.union1d(top_indices1, top_indices2)

    # Extract the scores from both arrays for the union of indices
    union_scores1 = array1[union_indices]
    union_scores2 = array2[union_indices]

    # Calculate the numerator and denominator for Ruzicka similarity
    numerator = np.sum(np.minimum(union_scores1, union_scores2))
    denominator = np.sum(np.maximum(union_scores1, union_scores2))

    unionstat = numerator / denominator

    # Get the intersection of indices from both top k lists
    inter_indices = np.intersect1d(top_indices1, top_indices2)

    # Extract the scores from both arrays for the union of indices
    inter_scores1 = array1[inter_indices]
    inter_scores2 = array2[inter_indices]

    # Calculate the numerator and denominator for Ruzicka similarity
    numerator2 = np.sum(np.minimum(inter_scores1, inter_scores2))
    denominator2 = np.sum(np.maximum(inter_scores1, inter_scores2))

    interstat = numerator2 / denominator2
    return interstat

def diff_inter_union(list1, list2, top_k=500):
    # Convert lists to numpy arrays for easier manipulation
    array1 = np.array(list1)
    array2 = np.array(list2)

    # Get the indices of the top k scores
    top_indices1 = np.argsort(-array1)[:top_k]
    top_indices2 = np.argsort(-array2)[:top_k]

    # Get the union of indices from both top k lists
    union_indices = np.union1d(top_indices1, top_indices2)

    # Extract the scores from both arrays for the union of indices
    union_scores1 = array1[union_indices]
    union_scores2 = array2[union_indices]

    diff_uion = np.sum(np.abs(union_scores1 - union_scores2))

    # Get the intersection of indices from both top k lists
    inter_indices = np.intersect1d(top_indices1, top_indices2)
    inter_scores1 = array1[inter_indices]
    inter_scores2 = array2[inter_indices]
    diff_inter = np.sum(np.abs(inter_scores1 - inter_scores2))
    return diff_inter/diff_uion

def weightedtautopk(list1, list2, top_k=500):
    # Convert lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)

    # Find the top k indices based on scores
    top_indices1 = np.argsort(-array1)[:top_k]
    top_indices2 = np.argsort(-array2)[:top_k]

    # Union of top indices from both lists
    union_indices = np.union1d(top_indices1, top_indices2)

    # Extract scores using the union indices in their original order
    union_scores1 = array1[union_indices]
    union_scores2 = array2[union_indices]

    # Calculate weighted Kendall's tau using scores directly
    tau, _ = weightedtau(union_scores1, union_scores2, rank=True)
    return tau


import matplotlib.pyplot as plt
def plot_metrics_eval(metrics_df_oneshotAOM):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('ROC_AUC', color=color)
    ax1.plot(metrics_df_oneshotAOM['ROC_AUC'], marker = 'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('AUPR', color=color)  # we already handled the x-label with ax1
    ax2.plot(metrics_df_oneshotAOM['AUPR'], marker = 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(metrics_df_oneshotAOM)))
    ax1.set_xticklabels(range(1, len(metrics_df_oneshotAOM) + 1))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # Create a figure for consecutive AOM
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    plt.figure(figsize=(10, 5))
    columns2 = ['JAC(MAX(S),MAX(S\d))']
    for column2,color1 in zip(columns2,colors):
        plt.plot(metrics_df_oneshotAOM[column2],marker='o',  linestyle='solid', color=color1, label=f'{column2}')
    plt.title('Between Consecutive Iterations')
    plt.legend()
    plt.xticks(range(len(metrics_df_oneshotAOM['JAC(MAX(S),MAX(S\d))'])), range(1, len(metrics_df_oneshotAOM['JAC(MAX(S),MAX(S\d))']) + 1))  # Set x-axis labels
    plt.show()

def plot_metrics_eval_mean(metrics_df_oneshotAOM,tau_df_copy):
    fig, ax1 = plt.subplots(figsize=(8, 4))

    color = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('ROC_AUC', color=color)
    ax1.plot(metrics_df_oneshotAOM['ROC_AUC'], marker = 'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('AUPR', color=color)  # we already handled the x-label with ax1
    ax2.plot(metrics_df_oneshotAOM['AUPR'], marker = 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(metrics_df_oneshotAOM)))
    ax1.set_xticklabels(range(1, len(metrics_df_oneshotAOM) + 1))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    tau_df_copy['Mean'] = tau_df_copy.mean(axis=1)
    tau_df_copy['Std'] = tau_df_copy.std(axis=1)

    # Create a figure for consecutive AOM
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    plt.figure(figsize=(8, 4))
    columns2 = ['JAC(MEAN(S),MEAN(S\d))']
    for column2,color1 in zip(columns2,colors):
        plt.plot(metrics_df_oneshotAOM[column2],marker='o',  linestyle='solid', color=color1, label=f'{column2}')
    plt.title('Between Consecutive Iterations')
    plt.legend()
    plt.xticks(range(len(metrics_df_oneshotAOM['JAC(MEAN(S),MEAN(S\d))'])), range(1, len(metrics_df_oneshotAOM['JAC(MEAN(S),MEAN(S\d))']) + 1))  # Set x-axis labels
    plt.show()
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    plt.figure(figsize=(8, 4))
    columns2 = ['Mean']
    for column2,color1 in zip(columns2,colors):
        plt.plot(tau_df_copy[column2],marker='o',  linestyle='solid', color=color1, label=f'{column2}')
    plt.title('Between Consecutive Iterations')
    plt.legend()
    plt.xticks(range(len(tau_df_copy['Mean'])), range(1, len(tau_df_copy['Mean']) + 1))  # Set x-axis labels
    plt.show()

def plot_metrics_SC2(metrics_df_oneshotAOM):
    # List of column pairs
    columns = ['Null_firstd_2', 'Null_Max(prev_ds)_2','Null_randomd_2']
    # List of colors
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    plt.figure(figsize=(10, 5))
    for column1,color1 in zip(columns,colors):
        plt.plot(metrics_df_oneshotAOM[column1],marker='o',  linestyle='solid', color=color1, label=f'{column1}')
    plt.title('Stopping Conditions: JAC(MAX(S),MAX(S\d)) - JAC(MAX(S_null),MAX(S\d))')
    plt.legend()
    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(range(len(metrics_df_oneshotAOM[column1])), range(1, len(metrics_df_oneshotAOM[column1]) + 1))  # Set x-axis labels
    plt.show()

    columns5 = ['JAC(MAX(Sprime),MAX(S\d))_p=first d', 'JAC(MAX(Sprime),MAX(S\d))_p=Max(previous ds)','JAC(MAX(Sprime),MAX(S\d))_p=random','JAC(MAX(S),MAX(S\d))']

    plt.figure(figsize=(10, 5))
    for column5,color1 in zip(columns5,colors):
        plt.plot(metrics_df_oneshotAOM[column5],marker='o',  linestyle='solid', color=color1, label=f'{column5}')
    plt.title('Stopping Conditions')
    plt.legend()
    plt.xticks(range(len(metrics_df_oneshotAOM[column1])), range(1, len(metrics_df_oneshotAOM[column1]) + 1))  # Set x-axis labels
    plt.show()

    plt.figure(figsize=(10, 5))
    columns4 = ['JAC_Average_beforeremoving','JAC_Std_beforeremoving']
    for column4,color1 in zip(columns4,colors):
        plt.plot(metrics_df_oneshotAOM[column4],marker='o',  linestyle='solid', color=color1, label=f'{column4}')
    plt.title('pairwise JAC mean and std at each iteration')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.xticks(range(len(metrics_df_oneshotAOM['JAC(AOM(S),AOM(S\d))'])), range(1, len(metrics_df_oneshotAOM['JAC(AOM(S),AOM(S\d))']) + 1))  # Set x-axis labels
    plt.show()

    plt.figure(figsize=(10, 5))
    columns3 = ['JAC(AOM)-JAC(MAX)']
    for column3,color1 in zip(columns3,colors):
        plt.plot(metrics_df_oneshotAOM[column3],marker='o',  linestyle='solid', color=color1, label=f'{column3}')
    plt.title('JAC(AOM) - JAC(MAX)')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.xticks(range(len(metrics_df_oneshotAOM['JAC(AOM(S),AOM(S\d))'])), range(1, len(metrics_df_oneshotAOM['JAC(AOM(S),AOM(S\d))']) + 1))  # Set x-axis labels
    plt.show()



def plot_metrics_SC1(metrics_df_oneshotAOM):
    # List of column pairs
    columns = ['Null_firstd_1', 'Null_AOM(prev_ds)_1','Null_randomd_1']
    # List of colors
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    plt.figure(figsize=(10, 5))
    for column1,color1 in zip(columns,colors):
        plt.plot(metrics_df_oneshotAOM[column1],marker='o',  linestyle='solid', color=color1, label=f'{column1}')
    plt.title('Stopping Conditions: JAC(AOM(S),MAX(S\d)) - JAC(AOM(S_null),MAX(S\d))')
    plt.legend()
    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(range(len(metrics_df_oneshotAOM[column1])), range(1, len(metrics_df_oneshotAOM[column1]) + 1))  # Set x-axis labels
    plt.show()

    columns5 = ['JAC(AOM(Sprime),MAX(S\d))_p=first d', 'JAC(AOM(Sprime),MAX(S\d))_p=AOM(previous ds)','JAC(AOM(Sprime),MAX(S\d))_p=random','JAC(AOM(S),MAX(S\d))']

    plt.figure(figsize=(10, 5))
    for column5,color1 in zip(columns5,colors):
        plt.plot(metrics_df_oneshotAOM[column5],marker='o',  linestyle='solid', color=color1, label=f'{column5}')
    plt.title('Stopping Conditions')
    plt.legend()
    plt.xticks(range(len(metrics_df_oneshotAOM[column1])), range(1, len(metrics_df_oneshotAOM[column1]) + 1))  # Set x-axis labels
    plt.show()
    plt.figure(figsize=(10, 5))
    columns4 = ['JAC_Average_beforeremoving','JAC_Std_beforeremoving']
    for column4,color1 in zip(columns4,colors):
        plt.plot(metrics_df_oneshotAOM[column4],marker='o',  linestyle='solid', color=color1, label=f'{column4}')
    plt.title('pairwise JAC mean and after')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.xticks(range(len(metrics_df_oneshotAOM['JAC(AOM(S),AOM(S\d))'])), range(1, len(metrics_df_oneshotAOM['JAC(AOM(S),AOM(S\d))']) + 1))  # Set x-axis labels
    plt.show()
    
    plt.figure(figsize=(10, 5))
    columns4 = ['JAC_Average_beforeremoving','JAC_Std_beforeremoving']
    for column4,color1 in zip(columns4,colors):
        plt.plot(metrics_df_oneshotAOM[column4],marker='o',  linestyle='solid', color=color1, label=f'{column4}')
    plt.title('pairwise JAC mean and std at each iteration')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.xticks(range(len(metrics_df_oneshotAOM['JAC(AOM(S),AOM(S\d))'])), range(1, len(metrics_df_oneshotAOM['JAC(AOM(S),AOM(S\d))']) + 1))  # Set x-axis labels
    plt.show()


def plot_JAC_2(tau_df_copy):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Mean', color=color)
    ax1.plot(tau_df_copy['Mean'], marker = 'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Std', color=color)  # we already handled the x-label with ax1
    ax2.plot(tau_df_copy['Std'], marker = 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(tau_df_copy)))
    ax1.set_xticklabels(range(1, len(tau_df_copy) + 1))
    plt.title('Mean & Std of JAC(MAX(S),MAX(S\d) for all d at each iteration')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_JAC_1(tau_df_copy):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Mean', color=color)
    ax1.plot(tau_df_copy['Mean'], marker = 'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Std', color=color)  # we already handled the x-label with ax1
    ax2.plot(tau_df_copy['Std'], marker = 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(tau_df_copy)))
    ax1.set_xticklabels(range(1, len(tau_df_copy) + 1))
    plt.title('Mean & Std of JAC(AOM(S),MAX(S\d) for all d at each iteration')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_ID_1(tau_df_copy):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Mean', color=color)
    ax1.plot(tau_df_copy['Mean'], marker = 'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Std', color=color)  # we already handled the x-label with ax1
    ax2.plot(tau_df_copy['Std'], marker = 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(tau_df_copy)))
    ax1.set_xticklabels(range(1, len(tau_df_copy) + 1))
    plt.title('Mean & Std of JAC_inlier(AOM(S),MAX(S\d) for all d at each iteration')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_ID_2(tau_df_copy):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Mean', color=color)
    ax1.plot(tau_df_copy['Mean'], marker = 'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Std', color=color)  # we already handled the x-label with ax1
    ax2.plot(tau_df_copy['Std'], marker = 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(tau_df_copy)))
    ax1.set_xticklabels(range(1, len(tau_df_copy) + 1))
    plt.title('Mean & Std of JAC_inlier(MAX(S),MAX(S\d) for all d at each iteration')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()