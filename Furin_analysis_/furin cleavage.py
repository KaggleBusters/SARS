import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.Seq import Seq
import matplotlib.pyplot as plt


def dna_to_amino_acid_makafim(dna_sequence, pattern, parent_linages):

    codons_qeue1 = []
    amino_acid_seq_dict = {}
    for dna_index, dna_char in enumerate(dna_sequence):
        if dna_char != '-':
            codons_qeue1.append([dna_char, dna_index])
            if len(codons_qeue1) == 3:
                dna_seq = Seq(codons_qeue1[0][0]+codons_qeue1[1][0]+codons_qeue1[2][0])
                amino_acid_seq = dna_seq.translate()
                key = (codons_qeue1[0][0]+codons_qeue1[1][0]+codons_qeue1[2][0], codons_qeue1[0][1], codons_qeue1[2][1])
                amino_acid_seq_dict[key] = amino_acid_seq
                codons_qeue1 = []
    furin_list = []
    amino_acid_seq_df = pd.DataFrame.from_dict(amino_acid_seq_dict, orient='index')
    amino_acid_seq_df = amino_acid_seq_df.reset_index()
    start_index = 0
    end_index = 0

    for ind, row in enumerate(amino_acid_seq_df.iterrows()):
        if str(row[1][0]) == 'R':
            for i in range(-1, 4, 1):
                furin_list.append(amino_acid_seq_df.iloc[ind+i, 1])
            if str(furin_list[1:]).replace("[", "").replace("]", "").replace("'", "").replace(",", "").replace(" ", "") == pattern[0]:
                start_index = amino_acid_seq_df.iloc[ind, 0][1]
                end_index = amino_acid_seq_df.iloc[ind+3, 0][2]
                pattern = pattern[0]
                break
            if str(furin_list[1:]).replace("[", "").replace("]", "").replace("'", "").replace(",", "").replace(" ", "") == pattern[1]:
                start_index = amino_acid_seq_df.iloc[ind, 0][1]
                end_index = amino_acid_seq_df.iloc[ind + 3, 0][2]
                pattern = pattern[1]
                break
            furin_list = []

    if 1 > len(furin_list):
        pattern = 'Null'
        furin_list = ['Null', '']
    return dna_sequence[start_index: end_index+1], start_index, end_index+1, pattern, furin_list[0]


def dna_to_amino_acid(dna_sequence):
    dna_seq = Seq(dna_sequence)
    amino_acid_seq = dna_seq.translate()
    return amino_acid_seq


def find_all(main_string, substring):
    start = 0
    while start < len(main_string):
        start = main_string.find(substring, start)
        if start == -1: break
        yield start
        start += 1


def find_furin_DNA_place(dna_seq, amino_acid_sequence, pattern):
    start = (amino_acid_sequence.find(pattern) * 3)
    finish = start + 15
    return dna_seq[start:finish]


def furin(spike_df, sequence_to_find_spike):
    # Define the sequence of interest for the Spike gene
    sequence_length_spike = len(sequence_to_find_spike)

    # Function to extract the first element from the "parent_lineages" column
    def extract_first_lineage(parent_lineage):
        return parent_lineage.split("/")[0]

    # Extract the first lineage from the "parent_lineages" column
    spike_df['first_lineage'] = spike_df['parent_lineages'].apply(extract_first_lineage)

    # Create a dictionary to store sequence positions for each first lineage
    sequence_positions_dict = {}

    # For each unique first lineage, determine the sequence position
    for lineage in spike_df['parent_lineages'].unique():
        lineage_sequence = spike_df[spike_df['parent_lineages'] == lineage].iloc[0]['cleaned_sequence']
        sequence_position = lineage_sequence.find(sequence_to_find_spike)
        if sequence_position != -1:
            sequence_positions_dict[lineage] = sequence_position

    # Function to extract sequence based on the position found in the first lineage
    def extract_relevant_sequence(row):
        if row['parent_lineages'] in sequence_positions_dict:
            pos = sequence_positions_dict[row['parent_lineages']]
            return row['cleaned_sequence'][pos:pos + sequence_length_spike]
        return None

    # Extract sequences based on the positions determined for each first lineage
    spike_df['relevant_sequence'] = spike_df.apply(extract_relevant_sequence, axis=1)

    # Drop rows where no relevant sequence was extracted
    filtered_spike_df = spike_df.dropna(subset=['relevant_sequence'])

    # Calculate conservation for each position in the extracted sequence
    conservation_relevant_sequence = []
    for i in range(sequence_length_spike):
        nucleotide_counts = filtered_spike_df['relevant_sequence'].str[i].value_counts()
        most_common_nucleotide_count = nucleotide_counts.max() if not nucleotide_counts.empty else 0
        conservation_relevant_sequence.append(most_common_nucleotide_count / len(filtered_spike_df))

    # Visualize the conservation rate
    plt.figure(figsize=(14, 6))
    plt.bar(range(sequence_length_spike), conservation_relevant_sequence, color='lightcoral')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Conservation Rate')
    plt.title('Conservation Rate of Each Position in the Spike Sequence for Relevant Lineages')
    plt.ylim(0, 1.1)  # to scale y-axis from 0 to 1
    plt.grid(axis='y')
    plt.show()

    return spike_df


def get_amino_seq(row):

    pattern = ["RRAR", "RRVR"]
    c_seq = row['sequence']
    parent_linages = row['parent_lineages']
    c_amino_seq, start_position, end_position, patt, leter_befor = dna_to_amino_acid_makafim(c_seq, pattern, parent_linages)
    list_one = [start_position, end_position]

    return c_amino_seq, list_one, patt, leter_befor


def get_amino_seq_clean(row):
    # if row['parent_lineages'] == 'XBB/XBB.2/XBB.2.1':
    #     print(0)
    pattern = ["RRAR"]
    c_seq = row['cleaned_sequence']
    c_amino_seq = dna_to_amino_acid(c_seq)
    n = list(find_all(c_amino_seq, pattern[0]))
    if len(n) == 0:
        n = []

    if len(n) > 1:
        print("FOUND TWO!")
        print(n)
        print(row['parent_lineages'])
    elif len(n) == 0:
        print("didnt found the furin cleavage")
    else:
        return pattern[0], find_furin_DNA_place(c_seq, c_amino_seq, pattern[0]), n[0]
    return "-1", "-1", "-1"


def calculate_similarity(s1, s2):
    # Count the number of matching characters
    match_count = sum(1 for i in range(len(s1)) if s1[i] == s2[i])
    # Calculate similarity
    similarity = match_count / len(s1)

    return similarity, match_count


def calc_furin_no_loc(df, wt_furin):
    pango_lineage_dict = {}
    df = df[df['furin_cleavage'] != '-1']
    for row_index, row in enumerate(df[['parent_lineages', 'sequence', 'furin_cleavage', 'furin_cleavage_positions']].iterrows()):
        parent_lin, clean_seq, furin_seq, furin_seq_pos = row[1][0], row[1][1], row[1][2], row[1][3]
        if furin_seq == 'A':
            continue
        start_pos, end_pos = int(furin_seq_pos.split(",")[0].replace('[', "")), int(furin_seq_pos.split(",")[1].replace(']', ""))
        temp_seq = []
        temp_seq_ind = []
        similarity_scores = {}
        for char_seq_index, char in enumerate(clean_seq):
            if char_seq_index > start_pos and char_seq_index < end_pos:
                temp_seq = []
                temp_seq_ind = []
                continue
            if char != '-':
                temp_seq.append(char)
                temp_seq_ind.append(char_seq_index)
            if len(temp_seq) == 12:
                window = clean_seq[temp_seq_ind[0]:temp_seq_ind[-1]+1]
                similarity_loc_precentage, similarity_count = calculate_similarity(window.replace('-', ''), wt_furin.replace('-', ''))#furin_seq.replace('-', ''))

                st = f'{temp_seq_ind[0]}-{temp_seq_ind[-1]}'
                similarity_scores[st] = [similarity_loc_precentage, similarity_count]
                temp_seq = []
                temp_seq_ind = []
        pango_lineage_dict[parent_lin] = similarity_scores

    return pango_lineage_dict


def hist_plot(data_dict):
    # Extract all similarity percentages
    all_percentages = [value[0] for inner_dict in data_dict.values() for value in inner_dict.values()]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_percentages, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Similarity Percentages")
    plt.xlabel("Similarity Percentage")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_unique_values_and_counts(df, column_name):
    # Get the specified column from the DataFrame
    column_data = df[column_name]

    # Calculate unique values and their counts
    unique_values = column_data.unique()
    value_counts = column_data.value_counts()

    # Create a dictionary where keys are unique values and values are their counts
    unique_values_dict = {value: count for value, count in zip(unique_values, value_counts)}

    # Create and display a bar graph
    plt.figure(figsize=(8, 6))
    bars = plt.bar(unique_values_dict.keys(), unique_values_dict.values(), color='skyblue')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title(f'Bar Graph of {column_name} Values and Counts')
    plt.xticks(rotation=0)

    # Add text labels above each column
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom')

    plt.show()


def count_strings(lst):
    string_count = {}  # Initialize an empty dictionary to store counts

    for item in lst:
        if item in string_count:
            string_count[item] += 1  # Increment the count if the string is already in the dictionary
        else:
            string_count[item] = 1  # Initialize the count to 1 if the string is not in the dictionary

    return string_count


def hist_plot_50precent_up(data_dict):
    # Extract all similarity percentages
    parent_with_only_50_up_dict = {}
    parent_with_only_50_up_count_dict = {}
    parent_with_only_50_up_1_4_dict = {}
    for ind1, (parent_lin, parent_v) in enumerate(data_dict.items()):
        list_50_up = {}
        for ind2, (seq_ind, v) in enumerate(parent_v.items()):
            value_p, value_c = v[0], v[0]
            if value_p > 0.5:
                list_50_up[seq_ind] =  value_p
        if len(list_50_up) != 0:
            parent_with_only_50_up_dict[parent_lin] = list_50_up
            parent_with_only_50_up_count_dict[parent_lin] = len(list_50_up)
            if len(list_50_up) != 2 and len(list_50_up) != 3:
                parent_with_only_50_up_1_4_dict[parent_lin] = [list_50_up, len(list_50_up)]

    # all_percentages = [value for inner_dict in parent_with_only_50_up_dict.values() for value in inner_dict.values()]
    all_percentages_counts = [inner_dict for inner_dict in parent_with_only_50_up_count_dict.values()]
    all_percentages = [value for inner_dict in parent_with_only_50_up_dict.values() for value in inner_dict.values()]
    all_percentages_keys = [value for inner_dict in parent_with_only_50_up_dict.values() for value in inner_dict.keys()]


    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_percentages, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Similarity Percentages X > 50%")
    plt.xlabel("Similarity Percentage")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Calculate the frequencies
    hist, bin_edges = np.histogram(all_percentages, bins=30)

    # Add text above each bar
    for i in range(len(hist)):
        if hist[i] != 0:
            plt.text(
                bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2,  # X position
                hist[i],  # Y position
                str(hist[i]),  # Text to display (frequency)
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                fontsize=8,  # Font size
                color='black',  # Text color
                weight='bold'  # Text weight
            )

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_percentages_keys, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of RRAR Seq Positions with Similarity Percentages X > 50%")
    plt.xlabel("RRAR Seq Positions")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Calculate the frequencies
    hist_dict = count_strings(all_percentages_keys)

    # Add text above each bar
    for i, (k, v) in enumerate(hist_dict.items()):
        plt.text(
           i,  # X position
            hist_dict[k] + 1,  # Y position
            str(hist_dict[k]),  # Text to display (frequency)
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize=8,  # Font size
            color='black',  # Text color
            weight='bold'  # Text weight
        )


    plt.figure(figsize=(10, 6))
    plt.hist(all_percentages_counts, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Number of RRAR Seq Positions with Similarity Percentages X > 50%")
    plt.xlabel("RRAR Seq Positions")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    hist, bin_edges = np.histogram(all_percentages_counts, bins=30)

    # Add text above each bar
    for i in range(len(hist)):
        if hist[i] != 0:
            plt.text(
                bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2,  # X position
                hist[i],  # Y position
                str(hist[i]),  # Text to display (frequency)
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                fontsize=8,  # Font size
                color='black',  # Text color
                weight='bold'  # Text weight
            )

    plt.show()


def heatmap_plot(data):
    # Determine the maximum length of sequences
    max_length = max(len(inner_dict) for inner_dict in data.values())

    # Create a 2D array for the heatmap with padding
    heatmap_data = np.zeros((len(data), max_length))
    for i, inner_dict in enumerate(data.values()):
        values = [value[0] for value in inner_dict.values()]
        heatmap_data[i, :len(values)] = values

    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Similarity Percentage'})
    plt.title("Heatmap of Similarity Percentages (Subset of Sequences)")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Sequence Index")
    plt.tight_layout()
    plt.show()


def cala_ave_sim_all_seq(data, N = 50):
    # Calculate average similarity percentages across all sequences for each position
    average_percentages = {position: np.mean([inner_dict.get(position, [0, 0])[0] for inner_dict in data.values()])
                           for position in data[list(data.keys())[0]]}

    # Get top N positions based on average similarity percentages
    top_positions = sorted(average_percentages, key=average_percentages.get, reverse=True)[:N]
    top_percentages = [average_percentages[pos] for pos in top_positions]

    # Plot the top N positions
    plt.figure(figsize=(15, 8))
    plt.bar(top_positions, top_percentages, color='dodgerblue')
    plt.title(f"Top {N} Positions with Highest Average Similarity Percentages")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Average Similarity Percentage")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #
    # df = pd.read_csv('lineages_added_seq_summary.tsv', sep='\t')
    # spike_df = df[df['gene'] == 'Spike']
    # spike_df = spike_df.reset_index()
    # spike_df.at[2233, 'sequence'] = spike_df.at[2032, 'sequence']
    # spike_df['furin_cleavage'], spike_df['furin_cleavage_positions'], spike_df['amino_seq'], spike_df['amino_char_befor_amino_seq']  = zip(*spike_df.apply(get_amino_seq, axis=1))

    # # # Remove "-" from the sequences in the dataframe
    # remove_or_false = False
    # if remove_or_false:
    #     spike_df['cleaned_sequence'] = spike_df['sequence'].str.replace("-", "")
    #     spike_df['furin_patern_clean'], spike_df['furin_cleavage_clean'], spike_df['starting_index_clean'] = zip(*spike_df.apply(get_amino_seq_clean, axis=1))
    # spike_df.to_csv("furin_2.csv")

    # df[df['parent_lineages'] == 'B/B.1/B.1.1/B.1.1.28/P.5']['sequence'].str[5425:5447]

    make_or_take = 'take'
    file_path = 'furin_within_seq_conservation.json'
    if make_or_take == 'make':
        df = pd.read_csv('furin_2.csv')
        data = calc_furin_no_loc(df,df[df['parent_lineages'] == 'B']['sequence'].values[0][5425:5447])

        # Specify the file path
        # Write the dictionary to the JSON file
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)

    if make_or_take == 'take':
        df = pd.read_csv('furin_2.csv')

        # Load the data from the uploaded JSON file
        with open(file_path, "r") as file:
            data = json.load(file)


    plot_unique_values_and_counts(df, 'amino_char_befor_amino_seq')
    plot_unique_values_and_counts(df, 'amino_seq')

    hist_plot(data)
    hist_plot_50precent_up(data)
    heatmap_plot(data)
    cala_ave_sim_all_seq(data)

    # import pandas as pd
    # import re
    #
    # # import tsv with lineage info (same as what I sent)
    # df = pd.read_csv('lineages_added_seq_summary.tsv', sep='\t')
    #
    # # remove non-spike sequences
    # spike_df = df[df['terminal_lineage'].str.contains('Spike')]
    #
    # # get the WT sequence
    # wt_msa_seq = list(df[df['terminal_lineage'] == 'B_Spike']['sequence'])[0]
    #
    # # remove dashes and obtain the WT nucleotide sequence of the PRRAR sequence - the index at the end is simply the index of the
    # # amino acid positions of PRRAR minu 1 then multiplied by 3 - so [(681 - 1)*3:(686-1)*3]
    # wt_prrar_nt_seq = list(df[df['terminal_lineage'] == 'B_Spike']['sequence'])[0].replace('-', '')[2040:2055]
    #
    # # create a regex search string - you can automate construction of this string as well, if you loop over all characters and add a "-*" after
    # # each one - I just manually made it for this example
    # regex_string = 'C-*C-*T-*C-*G-*G-*C-*G-*G-*G-*C-*A-*C-*G-*T'
    #
    # # search for the string in the multiple sequence alignment (msa) sequence - this will give you start (x.start()) and end (x.end()) indices,
    # # which you can use to find the PRRAR sequence (or mutated variants) in all of the sequences
    # x = re.search(regex_string, wt_msa_seq)
    #
    # # get all of the prrar sequences from all variants and add it as a column named "furin_cleavage_seqs" - I chose to remove "-" here because
    # # I want to translate them later but its probably not advised if we are interested in nucleotide conservation
    # spike_df['furin_cleavage_seqs'] = spike_df['sequence'].str[x.start():x.end()].str.replace("-", "")
    #
    #
    # # a simple translate function - you can get this from anywhere
    # def translate_sequence(sequence):
    #     codon_table = {
    #         "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    #         "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    #         "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    #         "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    #         "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    #         "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    #         "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    #         "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    #         "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    #         "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    #         "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    #         "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    #         "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    #         "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    #         "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    #         "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"
    #     }
    #
    #     protein_sequence = ""
    #     sequence = sequence.upper()  # Convert to uppercase for consistency
    #
    #     for i in range(0, len(sequence), 3):
    #         codon = sequence[i:i + 3]
    #         if codon in codon_table:
    #             protein_sequence += codon_table[codon]
    #         else:
    #             protein_sequence += "X"  # Placeholder for unknown codons
    #
    #     return protein_sequence
    #
    #
    # # translate the furin cleavage seqs and add as a new column
    # spike_df['translated_furin_seqs'] = spike_df['furin_cleavage_seqs'].apply(translate_sequence)
    #
    # # save as tsv
    # spike_df.to_csv('spike_df_furin.tsv', sep='\t', index=False)

