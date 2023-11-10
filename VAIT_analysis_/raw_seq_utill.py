import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_conservation_what_where(df, seq_fasta_type, loc_of_seq, seq_original):
    fasta_kind_df = df[df['fasta_kind'] == seq_fasta_type]
    dic_of_dif = {}
    # now we replace the makafim before serching !!
    for index_row, row in enumerate(fasta_kind_df[['raw_seq', 'gene_name']].iterrows()):
        seq, gen_name = row[1][0], row[1][1]
        seq_no_makaf = seq.replace('-', '')[loc_of_seq[0]: loc_of_seq[1]]
        dic_list_helper = []
        for char_index, (char_in_original, char_to_check) in enumerate(zip(seq_original, seq_no_makaf)):
            if char_in_original == char_to_check:
                continue
            else:
                dic_list_helper.append((loc_of_seq[0] + char_index, char_index, char_in_original, char_to_check))
        dic_of_dif[gen_name] = dic_list_helper
    return dic_of_dif


def calc_conservation_precentage(diff_dict, seq_to_check):
    value_sizes = {}
    conservation_percentages = {}
    seq_length = len(seq_to_check)

    for key, value in diff_dict.items():
        value_size = len(value)
        value_sizes[key] = value_size

        retention_percentage = value_size / seq_length
        conservation_percentages[key] = 1 - retention_percentage

    return conservation_percentages


def get_unique_values(input_dict):
    unique_values = {}
    for value in input_dict.values():
        if value in unique_values:
            unique_values[value] += 1
        else:
            unique_values[value] = 1
    return unique_values


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


def get_conservation_what_where_codons(df, seq_original):
    # Truncate sequences to make them divisible by 3
    seq_original = seq_original[:len(seq_original) - len(seq_original) % 3]

    df = df[df['conservation'] < 1]
    dic_of_dif = {}

    for index_row, row in df[['relevant_sequence_B', 'parent_lineages']].iterrows():
        seq, gen_name = row['relevant_sequence_B'], row['parent_lineages']

        # Truncate sequence to make it divisible by 3
        seq = seq[:len(seq) - len(seq) % 3]

        dic_list_helper = []

        # Split sequences into codons
        original_codons = [seq_original[i:i + 3] for i in range(0, len(seq_original), 3)]
        seq_codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]

        for codon_index, (codon_in_original, codon_to_check) in enumerate(zip(original_codons, seq_codons)):
            if codon_in_original == codon_to_check:
                continue
            else:
                # Using codon index multiplied by 3 to show the starting position of the codon
                dic_list_helper.append((codon_index * 3, codon_in_original, codon_to_check))
        dic_of_dif[gen_name] = dic_list_helper
    return dic_of_dif


def get_codons_checking_diffs(codons_str: tuple) -> dict:
    global amino_acid_to_codon_list
    what_was_was = codons_str[1]
    what_it_became = codons_str[2]
    global_index = codons_str[0]
    list_to_return = {}
    for key, values in amino_acid_to_codon_list.items():
        if what_was_was in values and what_it_became in values:
            print(key, what_was_was, "-->", what_it_became)
            for index, (letter_was, letter_will_be) in enumerate(zip(what_was_was, what_it_became)):
                if letter_was != letter_will_be:
                    list_to_return["{}_{}_{}".format(index, what_was_was, what_it_became)] = (
                        global_index, what_was_was, what_it_became, index + 1, letter_was, letter_will_be)
    return list_to_return


def get_dict_of_synonymos_changes_by_letter(full_dict: dict) -> dict:
    dict_of_synonimus = {}
    for key, values in full_dict.items():
        vals = []
        for value in values:
            val = get_codons_checking_diffs(value)
            if len(val) > 0:
                vals.append(val)
        if len(vals) > 0:
            dict_of_synonimus[key] = vals
    return dict_of_synonimus
