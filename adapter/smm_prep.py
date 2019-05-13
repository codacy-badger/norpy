"""
This contains the initial skecth of such a file
Wie genau funktioniert diese weighting Geschichte ?
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



def get_weighing_matrix(df_base, num_boots, num_agents_smm, is_store=False):
    """This function constructs the weighing matrix."""
    # Ensure reproducibility. What is df_base ? 
    np.random.seed(123)

    # Distribute clear baseline information.
    index_base = df_base.index.get_level_values('Identifier').unique()
    moments_base = get_moments(df_base)
    num_boots_max = num_boots * 2

    # Initialize counters to keep track of attempts.
    num_valid, num_attempts, moments_sample = 0, 0, []

    while True:

         try:
            sample_ids = np.random.choice(index_base, num_agents_smm, replace=False)
            moments_boot = get_moments(df_base.loc[(sample_ids, slice(None)), :])

            # We want to confirm that we have valid values for all required moments that we were
            # able to calculate on the observed dataset.
            for group in ['Choice Probability', 'Wage Distribution']:
                for period in moments_base[group].keys():
                    if period not in moments_boot[group].keys():
                        raise NotImplementedError

            moments_sample.append(moments_boot)

            num_valid += 1
            if num_valid == num_boots:
                break

            # We need to terminate attempts that just seem to not work.
            if num_attempts > num_boots_max:
                raise NotImplementedError("... too many samples needed for matrix")

        except NotImplementedError:
            continue

        num_attempts += 1

    # Construct the weighing matrix based on the sampled moments.
    stats = []
    for moments_boot in moments_sample:
        stats.append(moments_dict_to_list(moments_boot))

    moments_var = np.array(stats).T.var(axis=1)

    # We need to deal with the case that the variance for some moments is zero. This can happen
    # for example for the choice probabilities if early in the lifecycle nobody is working. This
    # will happen with the current setup of the final schooling moments, where we simply create a
    # grid of potential final schooling levels and fill it with zeros if not observed in the
    # data. We just replace it with the weight of an average moment.
    is_zero = moments_var <= 1e-10

    # TODO: As this only applies to the moments that are bounded between zero and one,
    #  this number comes up if I calculate the variance of random uniform variables.
    moments_var[is_zero] = 0.1

    if np.all(is_zero):
        raise NotImplementedError('... all variances are zero')
    if np.any(is_zero):
        print('... some variances are zero')

    weighing_matrix = np.diag(moments_var ** (-1))

    # Store information for further processing and inspection.
    if is_store:
        fname = 'weighing.respy.'
        np.savetxt(fname + 'txt', weighing_matrix, fmt='%35.15f')
        pkl.dump(weighing_matrix, open(fname + 'pkl', 'wb'))

    return weighing_matrix

