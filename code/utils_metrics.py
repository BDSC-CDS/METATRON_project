from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
from Levenshtein import distance
import math

################################## Step-order Evaluation ############################################
# Least Common Subsequence (LCS) score between two sequences (list of strings), normalized by the length of the reference sequence to penalize omissions.
def compute_lcs_score(ref, gen):
    matcher = SequenceMatcher(None, ref, gen)
    # print("Matching Blocks:", matcher.get_matching_blocks())
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    return lcs / len(ref) # Penalize omissions by dividing by the length of the reference sequence

def compute_strict_alignment(ref, gen):
    aligned = 0
    for r, g in zip(ref, gen):
        if r == g:
            aligned += 1
        else:
            break
    return {
        "aligned_steps": aligned,
        "total_ref_steps": len(ref),
        "score": round(aligned / len(ref), 3)
    }


################################## Content Evaluation ############################################
def empty_items(list1: list, list2: list, max_score=1):
    # Handle NA scenario
    if len(list1) == 0 and len(list2) == 0:
        return max_score  # Perfect match for both being empty 
    if len(list1) == 0 and len(list2) > 0 or len(list1) > 0 and len(list2) == 0:
        return 0  # One list is empty, assign 0
    if list1 is None or list2 is None or not isinstance(list1, list) or not isinstance(list2, list):
        return np.nan  # Invalid comparison
    return True 

# Mean or Max Token Set Ratio between the two lists of entities (to be computed for the lists associated with each label)
def compute_token_set_ratio(list1: list, list2: list, method="average"):
    output = empty_items(list1, list2, 100)
    if not(isinstance(output, bool)):
        return output
    else:
        # The maximum token set ratio for each item in list 1 when compare to each item in list 2
        scores = [
            max(fuzz.token_set_ratio(str(item1).lower(), str(item2).lower()) for item2 in list2) 
            for item1 in list1
        ]
        # Aggregate methods:
        # Choose how to extract one output for the entire field (list of strings)
        if method == "average":
            return np.mean(scores) if scores else np.nan
        elif method == "max":
            return np.max(scores) if scores else np.nan
        else:
            raise ValueError("Invalid method. Choose 'average' or 'max'")

# Mean or Max Normalized Levenshtein distance between the two lists of entities (to be computed for the lists associated with each label)
def compute_normalized_levenshtein(list1: list, list2: list, method="average"):
    output = empty_items(list1, list2)
    if not(isinstance(output, bool)):
        return output
    else:
        # Calculate Levenshtein distances normalized by max string length
        # Compute distance between each item in list1 and each one from list 2, then take the maximum similarity for each item in list1
        scores = [
            max(1 - (distance(item1, item2) / max(len(item1), len(item2), 1)) for item2 in list2)
            for item1 in list1
        ]
        # Aggregate scores
        if method == "average":
            return np.mean(scores) if scores else 0.0
        elif method == "max":
            return np.max(scores) if scores else 0.0
        else:
            raise ValueError("Invalid method. Choose 'average' or 'max'.")

# def compute_exact_match(list1, list2):
#     """
#     Compute exact match between two lists of strings.

#     Parameters:
#     ----------
#     list1 : list of str
#         The first list of strings to compare.
#     list2 : list of str
#         The second list of strings to compare.

#     Returns:
#     -------
#     int
#         1 if the lists are identical, 0 otherwise.
#     """
#     if list1 == list2:
#         return 1  # Exact match
#     return 0  # No match

# Change the exact match function to compute the percentage of exact matches between the two lists, instead of a binary score. 
# This allows for partial matches when the lists are not identical but share some common elements.
def compute_perc_exact_match(list1, list2):
    output = empty_items(list1, list2)
    if not(isinstance(output, bool)):
        return output
    else:        
        set1 = set(list1)    
        set2 = set(list2)
        intersection = set1.intersection(set2)
        return len(intersection)/len(set1) 

def choose_fun(metric:str, list1:list, list2:list):
    if metric == "LCS":
        output = compute_lcs_score(list1, list2)
    elif metric == "TSR":
        output = compute_token_set_ratio(list1, list2)
    elif metric == "Levenshtein":
        output = compute_normalized_levenshtein(list1, list2)
    elif metric == "Exact_Match_P":
        output = compute_perc_exact_match(list1, list2)
    return output
    
############################## Statistics ##########################
def statistics(list_score: list):
    statistics = {}
    statistics["mean"] = round(np.nanmean(list_score), 2)
    statistics["std"] = round(np.nanstd(list_score), 2)
    statistics["median"] = round(np.nanmedian(list_score), 2)
    statistics["percentiles"] = [round(np.nanpercentile(list_score, 25), 2), round(np.nanpercentile(list_score, 75), 2)]
    return statistics