import csv

from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def get_encoded_df(spike_sequences, encoding):

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


def top_n_longest_consecutive_sequences(data, parent_lineage, encoded_spike_df):

    # Extracting the parent lineage for the subset
    parent_lineage_subset = data['parent_lineages'].tolist()

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

    return filtered_subset, df_diff


def plot_vait_heatmap_per_tree(gene, encoding, filtered_subset, taxonomi_list):

    reverse_encoding = {v: k for k, v in encoding.items()}

    # Plotting the heatmap using the provided code with modifications to the colorbar
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(filtered_subset[filtered_subset.columns[:-1]][:-1], cmap="viridis",
                     cbar_kws={'label': 'Nucleotide'})

    # Setting the title and labels
    plt.xlabel("Position")
    plt.ylabel(f"Pango Lineages for parent {taxonomi_list[0]}")
    plt.yticks(range(len(taxonomi_list)), reversed(taxonomi_list))
    plt.title(f"Heatmap of the Positions {gene} Gene Sequences")

    # Adjusting the colorbar labels to show nucleotides
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(list(reverse_encoding.keys()))
    cbar.set_ticklabels(list(reverse_encoding.values()))

    plt.tight_layout()
    # plt.show()
    if not os.path.exists(f'plots/{gene}'):
        os.makedirs(f'plots/{gene}')

    plt.savefig(f"plots/{gene}/heat_map_nucli_{'_'.join(taxonomi_list)}_{gene}.png")
    plt.close()


def plot_varient_heatmap_per_tree(gene, filtered_subset, taxonomy_list):

    reverse_encoding = {1: 'a change',
                        2: 'no change'}

    # Define a custom colormap with red for change (1) and blue for no change (2)
    custom_cmap = ListedColormap(['red', 'blue'])

    # Plotting the heatmap using the custom colormap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(filtered_subset, cmap=custom_cmap, cbar_kws={'label': 'Nucleotide change'})

    # Setting the title and labels
    plt.xlabel("Position")
    plt.ylabel(f"Pango Lineages")
    plt.yticks(range(len(taxonomy_list)), reversed(taxonomy_list))
    plt.title(f"Heatmap of Change/No Change in the Positions {gene} Gene Sequences for parent {taxonomy_list[0]}")

    # Adjusting the colorbar labels to show nucleotides
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(list(reverse_encoding.keys()))
    cbar.set_ticklabels(list(reverse_encoding.values()))

    plt.tight_layout()
    # plt.show()
    if not os.path.exists(f'plots/{gene}'):
        os.makedirs(f'plots/{gene}')

    plt.savefig(f"plots/{gene}/heat_map_C_NO_C_{'_'.join(taxonomi_list)}_{gene}.png")
    plt.close()

def plot_variant_heatmap_per_tree_c(gene, encoding, filtered_subset, df_diff, taxonomy_list, precentage_of_change):
    reverse_encoding = {1: 'a change', 2: 'no change'}

    # Define a custom colormap with red for change (1) and blue for no change (2)
    custom_cmap = ListedColormap(['red', 'blue'])

    # Plotting the heatmap using the custom colormap
    plt.figure(figsize=(20, 18))  # Adjust the height for two plots
    ax1 = plt.subplot(2, 1, 1)  # Create the top subplot
    ax1 = sns.heatmap(df_diff, cmap=custom_cmap, cbar_kws={'label': 'Nucleotide change'})

    # Setting the title and labels for the top subplot
    ax1.set_xlabel("Position")
    ax1.set_ylabel(f"Pango Lineages")
    ax1.set_yticks(range(len(taxonomy_list)))
    ax1.set_yticklabels(reversed(taxonomy_list), rotation=45)
    ax1.set_title(f"Heatmap of Change/No Change in the Positions {gene} Gene Sequences for parent {taxonomy_list[0]} ({precentage_of_change*100:.2f}% has chaenged)")

    # Adjusting the colorbar labels to show nucleotides
    cbar1 = ax1.collections[0].colorbar
    cbar1.set_ticks(list(reverse_encoding.keys()))
    cbar1.set_ticklabels(list(reverse_encoding.values()))

    # Second heatmap
    reverse_encoding = {v: k for k, v in encoding.items()}

    ax2 = plt.subplot(2, 1, 2)  # Create the bottom subplot
    ax2 = sns.heatmap(filtered_subset[filtered_subset.columns[:-1]][:-1], cmap="viridis", cbar_kws={'label': 'Nucleotide'})

    # Setting the title and labels for the bottom subplot
    ax2.set_xlabel("Position")
    ax2.set_ylabel(f"Pango Lineages for parent {taxonomy_list[0]}")
    ax2.set_yticks(range(len(taxonomy_list)))
    ax2.set_yticklabels(reversed(taxonomy_list), rotation=45)
    ax2.set_title(f"Heatmap of the Positions {gene} Gene Sequences")

    # Adjusting the colorbar labels to show nucleotides
    cbar2 = ax2.collections[0].colorbar
    cbar2.set_ticks(list(reverse_encoding.keys()))
    cbar2.set_ticklabels(list(reverse_encoding.values()))

    plt.tight_layout()

    # Create the directory if it doesn't exist
    if not os.path.exists(f'plots/{gene}'):
        os.makedirs(f'plots/{gene}')

    plt.savefig(f"plots/{gene}/{gene}_combined_heatmaps_{'-'.join(taxonomy_list)}.png")
    plt.close()


# Function to find the maximum number of places
def max_places(lineage):
    parts = lineage.split('/')
    return len(parts)

def pot_pamgos_in_dummes(row, i):
    parts = row.parent_lineages.split('/')
    if len(parts) <= i:
        return 0
    else:
        return parts[i]

def get_just_end_taxonamy_rows(df, gene):

    # Apply the function to the 'parent_lineages' column and find the maximum
    value_to_remove = []
    for row in df.iterrows():
        parts = row[1].parent_lineages.split('/')
        if len(parts) > 1:
            for part in parts[:-1]:
                if part+"_"+str(gene) not in value_to_remove:
                    value_to_remove.append(part+"_"+str(gene))

    # Remove rows where the specified column contains values from the list
    df = df[~df['terminal_lineage'].isin(value_to_remove)]
    return df

def histogram_of_places_whom_changed(placeses_dict, gene):
    plac = [item for sublist in [places[1] for places in placeses_dict[gene]] for item in sublist]
    # plac = [places[2] for places in placeses_dict[gene]]
    # Create a histogram
    plt.hist(plac, bins=5, edgecolor='black', alpha=0.7)
    plt.xlabel(f'places indexes for {gene}')
    plt.ylabel('Frequency')
    plt.title('Histogram of places whom changed')

    # plt.show()
    plt.savefig(f"plots/{gene}/hist_of_places_changed.png")
    plt.close()


def combine_histograms(places_dict, gene):

    plt.figure(figsize=(15, 8))  # Set the figure size to accommodate two subplots
    # Subplot 1: Histogram of places whom changed
    plt.subplot(1, 2, 1)  # Create the first subplot
    plac_whom_changed = [item for sublist in [places[1] for places in places_dict[gene]] for item in sublist]
    plt.hist(plac_whom_changed, bins=len(set(plac_whom_changed)), edgecolor='black', alpha=0.7)
    plt.xlabel(f'places indexes for {gene}')
    plt.ylabel('Frequency')
    plt.title('Histogram of places whom changed')

    # Subplot 2: Histogram of places what changed
    plt.subplot(1, 2, 2)  # Create the second subplot
    plac_what_changed = [places[2] for places in places_dict[gene]]
    tuples_list = [tuple(inner_list) for inner_list in plac_what_changed]
    tuple_strings = ['->'.join(t) for t in tuples_list]
    # Count the frequency of each unique tuple string
    unique_tuple_strings, counts = zip(*[(ts, tuple_strings.count(ts)) for ts in set(tuple_strings)])
    # Create a bar graph
    plt.bar(unique_tuple_strings, counts, edgecolor='black', alpha=0.7)
    plt.xlabel(f'Which nucleoids were exchanged for which (the exchange from left to right) [{gene}]')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.title('Histogram of places what changed')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the combined image
    if not os.path.exists(f'plots/{gene}_all'):
        os.makedirs(f'plots/{gene}_all')

    plt.savefig(f"plots/{gene}_all/combined_histograms.png")
    plt.close()


def histogram_of_places_what_changed(placeses_dict, gene, encoding):

    plac = [places[2] for places in placeses_dict[gene]]
    mapped_plac = [[key for key, value in encoding.items() if value == val] for sublist in plac for val in sublist]

    # Create a histogram
    plt.hist(mapped_plac, bins=5, edgecolor='black', alpha=0.7)
    plt.xlabel(f'places indexes for {gene}')
    plt.ylabel('Frequency')
    plt.title('Histogram of places what changed')

    # plt.show()
    plt.savefig(f"plots/{gene}/hist_of_places_changed_what_nuc.png")
    plt.close()


# Reading the uploaded file
data = pd.read_csv("lineages_added_seq_summary.tsv", sep="\t")

# Encoding dictionary
encoding = {'-': 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5}
reverse_encoding = {v: k for k, v in encoding.items()}
for gene in data['gene'].unique():
    placeses_dict = {}
    if gene in ['ORF1ab', 'Spike']:
        placeses_dict[gene] = []
        parent_lin = 'B'

        gene_data_df = data[data['gene'] == gene]
        gene_data_df_copy = gene_data_df.copy()
        gene_data_df_copy['terminal_lineage_no'] = gene_data_df_copy['terminal_lineage'].apply(lambda x: x.split('_')[0])
        gene_data_df_filterd = get_just_end_taxonamy_rows(gene_data_df_copy, gene)

        for row in tqdm(gene_data_df_filterd.iterrows()):
            par_lin = row[1]['parent_lineages']

            if par_lin.__contains__('/'):
                taxonomi_list = par_lin.split('/')
                gene_data_df_ = gene_data_df_copy[gene_data_df_copy['terminal_lineage_no'].isin(taxonomi_list)]

                sequences = gene_data_df_.sequence.tolist()
                encoded_df = get_encoded_df(sequences, encoding)
                filtered_subset, df_diff = top_n_longest_consecutive_sequences(gene_data_df_, parent_lin, encoded_df)

                if not (df_diff == 2).all().all():
                    temp_ind_list = (df_diff == 2).all().index[~(df_diff == 2).all()].tolist()
                    temp_nuc_list = [filtered_subset[:-1][i] for i in temp_ind_list]
                    temp_nucli = [filtered_subset[:-1][i].tolist() for i in temp_ind_list][0]

                    temp_nucli_list = []
                    for el_in in range(len(temp_nucli)-1, 0, -1):
                        if el_in+1 == len(temp_nucli):
                            temp_nucli_list.append(reverse_encoding[int(temp_nucli[el_in])])
                        if temp_nucli[el_in] != temp_nucli[el_in-1]:
                            temp_nucli_list.append(reverse_encoding[int(temp_nucli[el_in-1])])

                    placeses_dict[gene].append([par_lin, temp_ind_list, temp_nucli_list, (len(temp_ind_list) / df_diff.shape[1], len(temp_ind_list) / (encoded_df.shape[1]-1)), set(temp_nucli), temp_nuc_list])
                    encoding_2 = {
                        'A': 2,
                        'C': 3,
                        'G': 4,
                        'T': 5}
                    plot_variant_heatmap_per_tree_c(gene, encoding_2, filtered_subset, df_diff, taxonomi_list, len(temp_ind_list) / df_diff.shape[1])

        combine_histograms(placeses_dict, gene)

        # Specify the CSV file path
        csv_file = f'plots/{gene}_all/my_dict.csv'
        # Write the list of lists to the CSV file
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write each sublist as a row in the CSV file
            for row in placeses_dict[gene]:
                writer.writerow(row)

        # histogram_of_places_whom_changed(placeses_dict, gene)
        # histogram_of_places_what_changed(placeses_dict, gene, encoding)