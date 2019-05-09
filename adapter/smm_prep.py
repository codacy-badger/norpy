"""
This contains the initial skecth of such a file
"""

import os

import numpy as np
import pandas as pd

from norpy.model_spec import get_model_obj



def update_model_spec(free_params,model_object,optim_paras):
    model_spec_dict = model_object.as_dict()
    for x in range(len(optim_paras)):
        model_spec_dict[optim_paras[x]] = free_params[x]
    out = get_model_obj(model_spec_dict)
    return out



def extract_relevant_results(array):
    df = pd.DataFrame({"period":array[:,2].reshape(len(array),1),
                       "choice":array[:,3].reshape(len(array),1),
                       "wages":array[:,4].reshape(len(array),1)})
    return df




def get_moments(result_array, is_store=False):
    """This function computes the moments based on a dataframe.
       Maybe we want another function to put the results of the simulation into
       a pd DataFrame altough would suggest that this is faster !
    """
    #Is des sinnvoll den hier reinzupacken? Könnte langsam werden
    analysis_df = extract_relevant_results(result_array)
    moments = OrderedDict()
    for group in ['Wage Distribution', 'Choice Probability']:
        moments[group] = OrderedDict()

    # We now add descriptive statistics of the wage distribution. Note that there might be
    # periods where there is no information available. In this case, it is simply not added to
    # the dictionary.

    info = analysis_df["wages"].copy().groupby('period').describe().to_dict()
    for period in sorted(analysis_df['period'].unique().tolist()):
        if pd.isnull(info['std'][period]):
            continue
        moments['Wage Distribution'][period] = []
        for label in ['mean', 'std']:
            moments['Wage Distribution'][period].append(info[label][period])

    # We first compute the information about choice probabilities. We need to address the case
    # that a particular choice is not taken at all in a period and then these are not included in
    # the dictionary. This cannot be addressed by using categorical variables as the categories
    # without a value are not included after the groupby operation.
    info = df['choice'].groupby('Period').value_counts(normalize=True).to_dict()
    for period in sorted(analysis_df['Period'].unique().tolist()):
        moments['Choice Probability'][period] = []
        for choice in range(1, 4):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments['Choice Probability'][period].append(stat)

    return moments

