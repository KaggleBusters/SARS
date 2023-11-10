import json
import numpy as np
import pandas as pd
import seaborn as sns
import raw_seq_utill as rsu
import matplotlib.pyplot as plt
import VAIT_extracter as ve

def read_data(path_to_data):
    df = pd.read_csv(path_to_data)
    return df

def connect_col_creation(x):
    return "{}_{}".format(x[1], x[0])

def VAIT_ORF1ab(df):

    # Filter the dataframe to only include rows corresponding to the gene "Spike"
    orf1ab_df = df[df['gene'] == 'ORF1ab']

    # Remove "-" from the sequences in the dataframe
    orf1ab_df['cleaned_sequence'] = orf1ab_df['sequence'].str.replace("-", "")

    # Define the sequence of interest for the Spike gene
    sequence_to_find = "GATGGTAAATCAAAATGTGAAGAATCATC"
    sequence_length_orf = len(sequence_to_find)

    # Function to extract the first element from the "parent_lineages" column
    def extract_first_lineage(parent_lineage):
        return parent_lineage.split("/")[0]

    # Extract the first lineage from the "parent_lineages" column
    orf1ab_df['first_lineage'] = orf1ab_df['parent_lineages'].apply(extract_first_lineage)

    # Create a dictionary to store sequence positions for each first lineage
    sequence_positions_dict = {}

    # For each unique first lineage, determine the sequence position
    for lineage in orf1ab_df['first_lineage'].unique():
        lineage_sequence = orf1ab_df[orf1ab_df['first_lineage'] == lineage].iloc[0]['cleaned_sequence']
        sequence_position = lineage_sequence.find(sequence_to_find)
        if sequence_position != -1:
            sequence_positions_dict[lineage] = sequence_position

    # Function to extract sequence based on the position found in the first lineage
    def extract_relevant_sequence(row):
        if row['first_lineage'] in sequence_positions_dict:
            pos = sequence_positions_dict[row['first_lineage']]
            return row['cleaned_sequence'][pos:pos + sequence_length_orf]
        return None

    # Extract sequences based on the positions determined for each first lineage
    orf1ab_df['relevant_sequence'] = orf1ab_df.apply(extract_relevant_sequence, axis=1)


    # Drop rows where no relevant sequence was extracted
    filtered_spike_df = orf1ab_df.dropna(subset=['relevant_sequence'])

    # Calculate conservation for each position in the extracted sequence
    conservation_relevant_sequence = []
    for i in range(sequence_length_orf):
        nucleotide_counts = filtered_spike_df['relevant_sequence'].str[i].value_counts()
        most_common_nucleotide_count = nucleotide_counts.max() if not nucleotide_counts.empty else 0
        conservation_relevant_sequence.append(most_common_nucleotide_count / len(filtered_spike_df))

    # Visualize the conservation rate
    plt.figure(figsize=(14, 6))
    plt.bar(range(sequence_length_orf), conservation_relevant_sequence, color='lightcoral')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Conservation Rate')
    plt.title('Conservation Rate of Each Position in the Spike Sequence for Relevant Lineages')
    plt.ylim(0, 1.1)  # to scale y-axis from 0 to 1
    plt.grid(axis='y')
    plt.show()

    return orf1ab_df

def VAIT_Spike(df):
    # Filter the dataframe to only include rows corresponding to the gene "Spike"
    spike_df = df[df['gene'] == 'Spike']

    # Remove "-" from the sequences in the dataframe
    spike_df['cleaned_sequence'] = spike_df['sequence'].str.replace("-", "")

    # Define the sequence of interest for the Spike gene
    sequence_to_find_spike = "TCTGCTTTACTAATGTCTATGCAGATT"
    sequence_length_spike = len(sequence_to_find_spike)

    # Function to extract the first element from the "parent_lineages" column
    def extract_first_lineage(parent_lineage):
        return parent_lineage.split("/")[0]

    # Extract the first lineage from the "parent_lineages" column
    spike_df['first_lineage'] = spike_df['parent_lineages'].apply(extract_first_lineage)

    # Create a dictionary to store sequence positions for each first lineage
    sequence_positions_dict = {}

    # For each unique first lineage, determine the sequence position
    for lineage in spike_df['first_lineage'].unique():
        lineage_sequence = spike_df[spike_df['first_lineage'] == lineage].iloc[0]['cleaned_sequence']
        sequence_position = lineage_sequence.find(sequence_to_find_spike)
        if sequence_position != -1:
            sequence_positions_dict[lineage] = sequence_position

    # Function to extract sequence based on the position found in the first lineage
    def extract_relevant_sequence(row):
        if row['first_lineage'] in sequence_positions_dict:
            pos = sequence_positions_dict[row['first_lineage']]
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

def calculate_similarity(s1, s2):
    # Count the number of matching characters
    match_count = sum(1 for i in range(len(s1)) if s1[i] == s2[i])

    # Calculate similarity
    similarity = match_count / len(s1)

    return similarity

def calcConservation(df, VAIT):

    conservations = {}
    for first_lineage in df['first_lineage']:
        d = df.copy()
        d = d[d['first_lineage'] == first_lineage]

        for i, row in d.iterrows():
            seq_to_check = row['relevant_sequence']
            conservations[row['parent_lineages']] = calculate_similarity(VAIT, seq_to_check)

    dd = pd.DataFrame.from_dict(conservations, orient='index')
    dd['parent_lineages'] = list(pd.DataFrame.from_dict(conservations, orient='index').index)

    df = pd.merge(df, dd, on='parent_lineages')
    return df

def get_conservation_what_where(df, seq_original):
    df = df[df['conservation'] < 1]
    dic_of_dif = {}
    # now we replace the makafim before serching !!
    for index_row, row in enumerate(df[['relevant_sequence', 'parent_lineages']].iterrows()):
        seq, gen_name = row[1][0], row[1][1]
        dic_list_helper = []
        for char_index, (char_in_original, char_to_check) in enumerate(zip(seq_original, seq)):
            if char_in_original == char_to_check:
                continue
            else:
                dic_list_helper.append((char_index, char_in_original, char_to_check))
        dic_of_dif[gen_name] = dic_list_helper
    return dic_of_dif

def plot_heat_map(js, name):
    positions = set()
    for mutations in js.values():
        for mutation in mutations:
            positions.add(mutation[0])
    positions = sorted(list(positions))
    df = pd.DataFrame(0, index=js.keys(), columns=positions)

    # Populating the DataFrame with mutation data
    for variant, mutations in js.items():
        for mutation in mutations:
            position = mutation[0]
            df.at[variant, position] = 1

    # Plotting the heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df, cmap='Blues', cbar=False)
    plt.title(f"Mutation Heatmap for {name}")
    plt.xlabel("Mutation Position")
    plt.ylabel("Variant")
    plt.show()

def plot_mutations(js, name):
    mutation_counts = {}

    for mutations in js.values():
        for mutation in mutations:
            position = mutation[0]
            original = mutation[1]
            mutated = mutation[2]
            key = f"{original}->{mutated}"
            if position not in mutation_counts:
                mutation_counts[position] = {}
            mutation_counts[position][key] = mutation_counts[position].get(key, 0) + 1

    # Transforming the mutation counts dictionary to a DataFrame
    df_mutations = pd.DataFrame(mutation_counts).fillna(0).transpose()

    # Plotting the heatmap for mutation counts
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_mutations, cmap='YlGnBu', annot=True, fmt='.0f', cbar=True)
    plt.title(f"Mutation Count Heatmap for {name}")
    plt.xlabel("Mutation Type (Original->Mutated)")
    plt.ylabel("Mutation position")
    plt.show()

def more_plots(js, name):
    # Re-creating the dictionary to hold mutation counts
    mutation_counts = {}

    for mutations in js.values():
        for mutation in mutations:
            position = mutation[0]
            original = mutation[1]
            mutated = mutation[2]
            key = f"{original}->{mutated}"
            if position not in mutation_counts:
                mutation_counts[position] = {}
            mutation_counts[position][key] = mutation_counts[position].get(key, 0) + 1

    # Re-transforming the mutation counts dictionary to a DataFrame
    df_mutations = pd.DataFrame(mutation_counts).fillna(0).transpose()
    df_ab = pd.DataFrame(0, index=js.keys(),
                         columns=sorted(list(set([mutation[0] for values in js.values() for mutation in values]))))

    # Populating the DataFrame with mutation data
    for variant, mutations in js.items():
        for mutation in mutations:
            position = mutation[0]
            df_ab.at[variant, position] = 1

    # Re-generating the visualizations

    # Visualization 1: Bar Plot of Mutation Counts
    plt.figure(figsize=(15, 5))
    mutation_types_counts = df_mutations.sum(axis=0).sort_values(ascending=False)
    mutation_types_counts.plot(kind='bar', color='skyblue')
    plt.title(f'Most Common Mutations Across All Positions for {name}')
    plt.xlabel('Mutation Type (Original->Mutated)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualization 2: Histogram of Mutations per Variant
    plt.figure(figsize=(15, 5))
    mutations_per_variant = df_ab.sum(axis=1)
    mutations_per_variant.plot(kind='hist', bins=20, color='salmon')
    plt.title(f'Histogram of Mutations per Variant for {name}')
    plt.xlabel('Number of Mutations')
    plt.ylabel('Number of Variants')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Visualization 3: Line Plot of Mutation Frequency by Position
    plt.figure(figsize=(15, 5))
    mutation_frequency_by_position = df_ab.sum(axis=0)
    mutation_frequency_by_position.plot(color='green')
    plt.title(f'Mutation Frequency by Position for {name}')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Number of Mutations')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Visualization 4: Pie Chart of Mutation Types
    mutation_types_percentage = mutation_types_counts / mutation_types_counts.sum()
    plt.figure(figsize=(10, 10))
    mutation_types_percentage.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title(f'Distribution of Mutation Types for {name}')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

def main():
    path = 'E:\SARS\SARS_only_raw_seq_and_mata_data.csv'
    FDA_df = pd.read_csv(r'E:\sars_summary_final_files\lineages_added_bicodons_master_counts2.tsv', sep='\t')

    df = read_data(path)
    df['connect_to_fda'] = df[['fasta_kind', 'gene_name']].apply(lambda x: connect_col_creation(x), axis=1)
    df2 = pd.merge(df, FDA_df[['parent_lineages','terminal_lineage' ]], how='left', left_on=['connect_to_fda'], right_on=['terminal_lineage'])

    seqs_to_check = ['GATGGTAAATCAAAATGTGAAGAATCATC', 'TCTGCTTTACTAATGTCTATGCAGATT']
    loc_of_seq = (7895, 7923)
    seq_fasta_types = ['ORF1ab', 'Spike']
    dif_dic_fasta_type = {}
    conservation_percentages_all_fastas = {}
    unique_values_all_fastas = {}

    for seq_fasta_type, seq_to_check in zip(seq_fasta_types, seqs_to_check):
        dif_dic = rsu.get_conservation_what_where(df, seq_fasta_type, loc_of_seq, seq_to_check)
        dif_dic_fasta_type[seq_fasta_type] = dif_dic

        conservation_percentages = rsu.calc_conservation_precentage(dif_dic, seq_to_check)
        conservation_percentages_all_fastas[seq_fasta_type] = conservation_percentages

        unique_values = rsu.get_unique_values(conservation_percentages)
        unique_values_all_fastas[seq_fasta_type] = unique_values

    # Load the uploaded TSV file into a DataFrame
    load_from_TSV = False
    if load_from_TSV:
        df = pd.read_csv('lineages_added_seq_summary.tsv', sep='\t')

        ab_df = VAIT_ORF1ab(df)
        spike_df = VAIT_Spike(df)

        ab_df = calcConservation(ab_df, "GATGGTAAATCAAAATGTGAAGAATCATC")
        spike_df = calcConservation(spike_df, "TCTGCTTTACTAATGTCTATGCAGATT")

        ab_df.to_csv("ab_VAIT_conservation.csv")
        spike_df.to_csv("spike_VAIT_conservation.csv")

        ab_df = pd.read_csv("ab_VAIT_conservation.csv")
        spike_df = pd.read_csv("spike_VAIT_conservation.csv")

        ab_dict = get_conservation_what_where(ab_df, "GATGGTAAATCAAAATGTGAAGAATCATC")
        spike_dict = get_conservation_what_where(spike_df, "TCTGCTTTACTAATGTCTATGCAGATT")
    else:
        with open('spike.json', 'r') as file:
            spike_data = json.load(file)

        with open('ab.json', 'r') as file:
            ab_data = json.load(file)

    plot_heat_map(ab_data, "ORF1ab")
    plot_heat_map(spike_data, "Spike")
    plot_mutations(ab_data, "ORF1ab")
    plot_mutations(spike_data, "Spike")

    more_plots(spike_data, "Spike")
    more_plots(ab_data, "ORF1ab")

def main_extra_VAIT():
    # Reading the uploaded file
    data = pd.read_csv("lineages_added_seq_summary.tsv", sep="\t")
    # data = data.iloc[:200, :]
    # Encoding dictionary
    encoding = {
        '-': 1,
        'A': 2,
        'C': 3,
        'G': 4,
        'T': 5
    }
    # gene = 'Spike'
    parent_lin = 'B'
    top_n = 10

    for gene in data['gene'].unique():
        if gene == 'E':
            encoded_df = ve.get_encoded_df(data, encoding, gene)
            top_N_sequences, filtered_subset, dif_df = ve.top_n_longest_consecutive_sequences(data, gene, parent_lin, encoded_df, n=top_n)

            spike_sequences = data[data['gene'] == gene].sequence.tolist()
            ve.plot_varient_heatmap_per_tree(gene, dif_df)
            ve.plot_vait_heatmap(top_N_sequences, gene, encoding, filtered_subset, spike_sequences)


if __name__ == '__main__':
    # main()
    main_extra_VAIT()
