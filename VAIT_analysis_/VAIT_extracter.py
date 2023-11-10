import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def get_encoded_df(data, encoding, gene='Spike'):

    spike_sequences = data[data['gene'] == gene].sequence.tolist()

    # Determine the maximum sequence length for Spike gene sequences
    spike_max_len = max(len(seq) for seq in spike_sequences)

    # Creating an empty array with dimensions (number of Spike sequences x maximum sequence length)
    encoded_spike_sequences = np.zeros((len(spike_sequences), spike_max_len), dtype=int)

    # Populating the array with encoded values
    for i, seq in enumerate(spike_sequences):
        for j, char in enumerate(seq):
            encoded_spike_sequences[i, j] = encoding.get(char, 0)  # 0 for any character not in the encoding dictionary

    # Converting to DataFrame for visualization
    encoded_spike_df = pd.DataFrame(encoded_spike_sequences)

    return encoded_spike_df


def top_n_longest_consecutive_sequences(data, gene, parent_lineage, encoded_spike_df, n):

    # Extracting the parent lineage for the subset
    parent_lineage_subset = data[data['gene'] == gene]['parent_lineages'].tolist()

    # Adding the parent lineage to the subset dataframe
    encoded_spike_df['parent_lineage'] = parent_lineage_subset

    # Filtering rows where parent lineage contains 'B'
    subset_b_lineage = encoded_spike_df[encoded_spike_df['parent_lineage'] == parent_lineage]

    # Identifying columns where any value in the B lineage subset is 1
    columns_to_drop = subset_b_lineage.columns[(subset_b_lineage == 1).any()]

    # Dropping those columns from the main subset DataFrame
    filtered_subset = encoded_spike_df.drop(columns=columns_to_drop)

    # Calculating variance for each column, excluding 'parent_lineage'
    variance = filtered_subset.drop(columns='parent_lineage').var()

    # Adding the variance as the last row
    filtered_subset.loc['variance'] = variance
    filtered_subset.loc['variance', 'parent_lineage'] = 'Variance'

    filtered_subset.columns = list(range(1, len(filtered_subset.columns) + 1))

    # Create df_diff with all values initialized to 2
    df_diff = pd.DataFrame(2, index=filtered_subset[filtered_subset.columns[:-1]][:-1].index,
                           columns=filtered_subset[filtered_subset.columns[:-1]][:-1].columns)

    # Check for changes and set values to 1 where changes occur
    for column in filtered_subset[filtered_subset.columns[:-1]][:-1].columns:
        if filtered_subset[filtered_subset.columns[:-1]][:-1][column].nunique() > 1:
            df_diff[column] = 1

    # Identifying columns with variance of 0
    zero_variance_columns = variance[variance == 0].index.tolist()

    # Finding the top 3 longest consecutive sequences
    consecutive_sequences = []
    current_sequence = []

    for i in range(len(zero_variance_columns) - 1):
        current_sequence.append(zero_variance_columns[i])
        if zero_variance_columns[i+1] - zero_variance_columns[i] != 1:
            consecutive_sequences.append(current_sequence)
            current_sequence = []

    # Adding the last sequence if it wasn't added
    if current_sequence:
        consecutive_sequences.append(current_sequence)

    # Sorting sequences by length and getting the top 3
    top_n_sequences = sorted(consecutive_sequences, key=len, reverse=True)[:n]
    # print(top_3_sequences)

    sequence_dict = {'index': [], 'sequence': [], 'position': [], 'len': []}
    for i in range(len(top_n_sequences)):
        sequence_dict['index'].append(i)
        sequence_dict['sequence'].append(
            data[data['gene'] == gene].sequence.tolist()[0][top_n_sequences[i][0]:top_n_sequences[i][-1] + 1])
        sequence_dict['position'].append((top_n_sequences[i][0], top_n_sequences[i][-1] + 1))
        sequence_dict['len'].append(
            len(data[data['gene'] == gene].sequence.tolist()[0][top_n_sequences[i][0]:top_n_sequences[i][-1] + 1]))

    sequences_df = pd.DataFrame.from_dict(sequence_dict)

    if not os.path.exists(f'plots/{gene}_all'):
        os.makedirs(f'plots/{gene}_all')

    sequences_df.to_csv(f"plots/{gene}_all/potential_VAITS_WT.csv")
    filtered_subset.to_csv(f"plots/{gene}_all/clean_sequence_encoded_WT.csv")

    return top_n_sequences, filtered_subset, df_diff


def plot_vait_heatmap(top_N_sequences, gene, encoding, filtered_subset, spike_sequences):

    sequence_list = []
    for i in range(len(top_N_sequences)):
        sequence_list.append(spike_sequences[0][top_N_sequences[i][0]:top_N_sequences[i][-1]+1])
        print(f'Seq {i} start from {top_N_sequences[i][0]} to {top_N_sequences[i][-1]} ({len(sequence_list[i])})')
        print(f'The seq: {sequence_list[i]}')

    reverse_encoding = {v: k for k, v in encoding.items()}

    # Plotting the heatmap using the provided code with modifications to the colorbar
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(filtered_subset[filtered_subset.columns[:-1]][:-1], cmap="viridis",
                     cbar_kws={'label': 'Nucleotide'})

    # Setting the title and labels
    plt.xlabel("Position")
    plt.ylabel("Parent lineage Index")
    plt.title(f"Heatmap of the Positions {gene} Gene Sequences")

    # Adjusting the colorbar labels to show nucleotides
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(list(reverse_encoding.keys()))
    cbar.set_ticklabels(list(reverse_encoding.values()))

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"plots/{gene}_all/heat_map_nucli_{gene}.png")

def plot_varient_heatmap_per_tree(gene, filtered_subset):

    reverse_encoding = {2: 'no change',
                        1: 'a change'}

    # Define a custom colormap with red for change (1) and blue for no change (2)
    custom_cmap = ListedColormap(['red', 'blue'])

    # Assuming filtered_subset is your data and taxonomy_list is your list of labels

    # Plotting the heatmap using the custom colormap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(filtered_subset, cmap=custom_cmap, cbar_kws={'label': 'Nucleotide change'})

    # Setting the title and labels
    plt.xlabel("Position")
    plt.ylabel(f"Pango Lineages")
    plt.title(f"Heatmap of Change/No Change in the Positions {gene} Gene Sequences")

    # Adjusting the colorbar labels to show nucleotides
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(list(reverse_encoding.keys()))
    cbar.set_ticklabels(list(reverse_encoding.values()))

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"plots/{gene}_all/heat_map_C_NO_C_{gene}.png")