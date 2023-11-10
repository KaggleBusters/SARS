import os
import itertools
import numpy as np
import pandas as pd
import datetime as dt
from Bio import SeqIO
from tqdm import tqdm
from collections import Counter

class Fasta_Handler():

    def __init__(self, df):
        self.main_df = df

    def get_sum_of_dict_values(self, dictionary: dict) -> int:
        sum = 0
        for value in dictionary.values():
            sum += value
        return sum

    def get_meta_data_from_file(self, filename):

        # gene_name
        gene_name = filename.split('/')[-1].split('_')[0]
        # fasta_kind
        fasta_kind = filename.split('/')[-1].split('_')[1]
        return gene_name, fasta_kind

    def extract_all_seq(self, sequence):
        doubles = [str(sequence[i:len(sequence) - 1]) for i in range(0, len(sequence), len(sequence))]
        return doubles

    def extract_doubles(self, sequence):
        doubles = [str(sequence[i:i + 2]) for i in range(0, len(sequence) - 2, 1)]
        return doubles

    def extract_codons(self, sequence):
        codons = [str(sequence[i:i + 3]) for i in range(0, len(sequence) - 3, 3)]
        return codons


    def extract_bio_codons(self, sequence):
        bio_codons = [str(sequence[i:i + 6]) for i in range(0, len(sequence) - 6, 3)]
        return bio_codons


    def remove_makaf_from_counter_or_dict(self, counter_val, letter):
        counter_to_subtract = Counter()
        counter_of_samples_dropped = 0
        for counter_item in counter_val:
            if letter in str(counter_item):
                counter_of_samples_dropped += counter_val[counter_item]
                counter_to_subtract[counter_item] = counter_val[counter_item]
        return counter_val - counter_to_subtract, counter_of_samples_dropped


    def normalized_counter_samples(self, counter_val, total_samples):
        for counter_item in counter_val:
            counter_val[counter_item] = counter_val[counter_item] / total_samples
        return counter_val


    def get_all_combinations_for_features_doubles(self, alphabets):

        lst = []
        for i in alphabets:
            for j in alphabets:
                lst.append(i + j)
        return lst


    def entropy_1(self):
        self.main_df['entropy1'] = -1 * (
                    self.main_df[['A', 'G', 'T', 'C']] * np.log2(self.main_df[['A', 'G', 'T', 'C']])).sum(axis=1)


    def entropy_2(self):
        c2 = self.get_all_combinations_for_features_doubles(['A', 'G', 'T', 'C'])
        self.main_df['entropy2'] = -1 * (self.main_df[c2] * np.log2(self.main_df[c2])).sum(axis=1)


    def entropy_3(self):

        column_names = self.main_df.columns[self.main_df.columns.str.len() == 3].tolist()
        self.main_df['entropy3'] = -1 * (self.main_df[column_names] * np.log2(self.main_df[column_names])).sum(axis=1)


    def generate_bicodons(self):
        bases = "AGTC"
        bicodons = [''.join(codon) for codon in itertools.product(bases, repeat=6)]
        return bicodons


    def entropy_6(self):
        # cant add this because the df has not have all the codons comnbinations in his columns
        c3 = self.generate_bicodons()

        column_names = self.main_df.columns[self.main_df.columns.isin(c3)].tolist()
        self.main_df['entropy6'] = -1 * (self.main_df[column_names] * np.log2(self.main_df[column_names])).sum(axis=1)


    def get_all_combinations_for_features_codons(self, alphabets):

        lst = []
        for i in alphabets:
            for j in alphabets:
                for x in alphabets:
                    lst.append(i + j + x)
        return lst


    def get_all_combinations_for_features(self, alphabets):

        lst = []
        for i in alphabets:
            for j in alphabets:
                for x in alphabets:
                    lst.append(i + j + x)
                    for z in alphabets:
                        lst.append(i + j + x + z)
        return lst


    def main_extract_fasta_from_dir(self, directory):
        # iterate over all directories
        for directory_name in os.listdir(directory):
            codons_counter = Counter()
            doubles_counter = Counter()
            ones_counter = Counter()
            bi_codon_counter = Counter()

            country_counter = Counter()
            continent_counter = Counter()

            sequences = []
            sequences_meta_data = []
            total_len_of_sequences = 0
            sequence_counter = 0
            lenght_of_sequence = 0
            global_date = None

            # iterate over all fasta files in directory
            if directory_name.startswith('.') == False:
                for filename in tqdm(os.listdir(directory + '{}'.format(directory_name))):
                    if filename.endswith('.fasta'):
                        # Read the sequences from the FASTA file
                        total_len_of_sequences += len(
                            list(SeqIO.parse(directory + directory_name + '/' + filename, "fasta")))

                        for index, sequence in enumerate(
                                SeqIO.parse(directory + directory_name + '/' + filename, "fasta")):
                            country = {str(sequence.description.split(" ")[2]): 1}
                            continent = {str(sequence.description.split(" ")[3]): 1}
                            country_counter.update(country)
                            continent_counter.update(continent)

                            str_date = sequence.description.split(" ")[1]
                            if global_date is None:
                                if str_date.split("-")[-1] == "00":
                                    str_date = str_date.replace("00", "01")
                                if str_date.split("-")[-2] == "00":
                                    str_date = str_date.replace("00", "01")
                                global_date = dt.datetime.strptime(str_date, '%Y-%m-%d').date()

                            else:
                                if str_date.split("-")[-1] == "00":
                                    str_date = str_date.replace("00", "01")
                                if str_date.split("-")[-2] == "00":
                                    str_date = str_date.replace("00", "01")
                                local_date = dt.datetime.strptime(str_date, '%Y-%m-%d').date()
                                if global_date > local_date:
                                    global_date = local_date

                            if index == 0:
                                lenght_of_sequence += len(str(sequence.seq).replace("-", ""))
                                sequences.append(str(sequence.seq))
                                sequences_meta_data.append(sequence.description)

                                clade = sequences_meta_data[0].split(" ")[5]
                                variant = sequences_meta_data[0].split(" ")[6]

                                # extract ones
                                ones_counter.update(Counter(sequence.seq))
                                # extract doubles
                                doubels = self.extract_doubles(sequence.seq)
                                doubles_counter.update(Counter(doubels))

                                # Extract codons
                                codons = self.extract_codons(sequence.seq)
                                codons_counter.update(Counter(codons))

                                bicodon = self.extract_bio_codons(sequence.seq)
                                bi_codon_counter.update(bicodon)

                                # total_len_of_sequences.append(len(sequence))
                                sequence_counter += 1

                total_ones_in_gene = self.get_sum_of_dict_values(ones_counter)
                total_doubles_in_gene = self.get_sum_of_dict_values(doubles_counter)
                total_codons_in_gene = self.get_sum_of_dict_values(codons_counter)
                total_bi_codons_in_gene = self.get_sum_of_dict_values(bi_codon_counter)

                no_makafs_ones_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(ones_counter, '-')
                ones_updated_counter = self.normalized_counter_samples(no_makafs_ones_counter,
                                                                       int(total_ones_in_gene - samples_dropped))

                no_makafs_doubles_counter, samples_dropped = self.remove_makaf_from_counter_or_dict_v2(doubles_counter,
                                                                                                       '-', 2)
                doubles_updated_counter = self.normalized_counter_samples(no_makafs_doubles_counter,
                                                                          int(total_doubles_in_gene - samples_dropped))

                no_makafs_codons_counter, samples_dropped = self.remove_makaf_from_counter_or_dict_v2(codons_counter,
                                                                                                      '-', 3)
                codons_updated_counter = self.normalized_counter_samples(no_makafs_codons_counter,
                                                                         int(total_codons_in_gene - samples_dropped))

                no_makafs_bi_codons_counter, samples_dropped = self.remove_makaf_from_counter_or_dict_v2(
                    bi_codon_counter, '-', 6)
                bi_codons_updated_counter = self.normalized_counter_samples(no_makafs_bi_codons_counter,
                                                                            int(total_bi_codons_in_gene - samples_dropped))

                ones_updated_counter.update(doubles_updated_counter)
                ones_updated_counter.update(codons_updated_counter)
                ones_updated_counter.update(bi_codons_updated_counter)

                # all_codons_dict = dict(codons_updated_counter+doubles_updated_counter+ones_updated_counter)
                all_codons_dict = ones_updated_counter
                all_codons_dict['gene_name'] = directory_name
                # counting the total sequnces in every fasta file for each gene
                all_codons_dict['total_sequences'] = total_len_of_sequences
                # all the countries sampled of every fasta file for each gene
                all_codons_dict['countries'] = country_counter
                # all the continents sampled of every fasta file for each gene
                all_codons_dict['continents'] = continent_counter
                # the first sequnce lenght of each fasta file summed for each gene
                all_codons_dict['sequences_lenght'] = lenght_of_sequence
                all_codons_dict['first_date'] = global_date
                all_codons_dict['variant'] = variant
                all_codons_dict['clade'] = clade

                if (len(self.main_df) == 0):
                    self.main_df = self.main_df.from_dict([all_codons_dict])
                else:
                    self.main_df.loc[len(self.main_df)] = pd.Series(all_codons_dict)
        print(f"Finish extract {directory}")


    def main_extract_fasta_from_dir__(self, directory):
        # iterate over all directories
        for directory_name in tqdm(os.listdir(directory)):
            print(f"\nStart extract {directory_name}")
            # iterate over all fasta files in directory
            if len(os.listdir(directory + '/{}'.format(directory_name))) < 11:
                print(f"dir {directory_name} containe LESS then 11 fastas!!!")
            for filename in os.listdir(directory + '/{}'.format(directory_name)):
                if filename.endswith('.fasta'):
                    # Read the sequences from the FASTA file
                    sequence_counter = 0
                    for seq_index, record in enumerate(
                            SeqIO.parse(directory + directory_name + '/' + filename, "fasta")):
                        # if seq_index == 0:
                        meta_data = str(record.description).split(" ")
                        codons_counter = Counter()
                        doubles_counter = Counter()
                        ones_counter = Counter()
                        sequence_counter += 1

                        # extract ones
                        ones_counter.update(Counter(str(record.seq)))

                        # extract doubles
                        doubels = self.extract_doubles(str(record.seq))
                        doubles_counter.update(Counter(doubels))

                        # Extract codons
                        codons = self.extract_codons(str(record.seq))
                        codons_counter.update(Counter(codons))

                        total_ones_in_fasta = len(record.seq)
                        total_doubles_in_fasta = len(doubels)
                        total_codons_in_fasta = len(codons)

                        if self.counting_makafs:
                            no_makafs_ones_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                                ones_counter, '-')
                            ones_updated_counter = self.normalized_counter_samples(no_makafs_ones_counter,
                                                                                   int(total_ones_in_fasta - samples_dropped))

                            no_makafs_doubles_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                                doubles_counter, '-')
                            doubles_updated_counter = self.normalized_counter_samples(no_makafs_doubles_counter,
                                                                                      int(total_doubles_in_fasta - samples_dropped))

                            no_makafs_codons_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                                codons_counter, '-')
                            codons_updated_counter = self.normalized_counter_samples(no_makafs_codons_counter,
                                                                                     int(total_codons_in_fasta - samples_dropped))
                        else:
                            ones_updated_counter = self.normalized_counter_samples(ones_counter, total_ones_in_fasta)
                            doubles_updated_counter = self.normalized_counter_samples(doubles_counter,
                                                                                      total_doubles_in_fasta)
                            codons_updated_counter = self.normalized_counter_samples(codons_counter,
                                                                                     total_codons_in_fasta)
                        all_codons_dict = dict(codons_updated_counter + doubles_updated_counter + ones_updated_counter)

                        # add_meta_data_to_dict
                        gene_name, fasta_kind = self.get_meta_data_from_file(filename)
                        all_codons_dict['gene_name'] = gene_name
                        all_codons_dict['fasta_kind'] = fasta_kind
                        all_codons_dict['ID'] = meta_data[0]
                        all_codons_dict['date'] = meta_data[1]
                        all_codons_dict['country'] = meta_data[2]
                        all_codons_dict['continent'] = meta_data[3]
                        all_codons_dict['?'] = meta_data[5]
                        all_codons_dict['variant'] = meta_data[-1]

                        if (len(self.main_df) == 0):
                            self.main_df = self.main_df.from_dict([all_codons_dict])
                        else:
                            self.main_df.loc[len(self.main_df)] = pd.Series(all_codons_dict)
            print(f"Finish extract {directory_name}")


    def main_extract_fasta_from_dir_by_kind(self, directory):
        # iterate over all directories
        for directory_name in tqdm(os.listdir(directory)):
            codons_counter = Counter()
            doubles_counter = Counter()
            ones_counter = Counter()
            bi_codon_counter = Counter()
            country_counter = Counter()
            continent_counter = Counter()

            sequences_meta_data = []
            global_date = None

            # iterate over all fasta files in directory
            if directory_name.startswith('.') == False:
                for filename in tqdm(os.listdir(directory + '{}'.format(directory_name))):
                    if filename.endswith('.fasta'):
                        # Read the sequences from the FASTA file
                        most_common_seq_counter = Counter()
                        temp_sequence = None
                        for index, sequence in enumerate(
                                SeqIO.parse(directory + directory_name + '/' + filename, "fasta")):
                            country = {str(sequence.description.split(" ")[2]): 1}
                            continent = {str(sequence.description.split(" ")[3]): 1}
                            country_counter.update(country)
                            continent_counter.update(continent)

                            all_seq_seq = self.extract_all_seq(sequence.seq)
                            most_common_seq_counter.update(all_seq_seq)

                            if index == 0:
                                temp_sequence = sequence.description

                            str_date = sequence.description.split(" ")[1]
                            if global_date is None:
                                if str_date.split("-")[-1] == "00":
                                    str_date = str_date.replace("00", "01")
                                if str_date.split("-")[-2] == "00":
                                    str_date = str_date.replace("00", "01")
                                global_date = dt.datetime.strptime(str_date, '%Y-%m-%d').date()

                            else:
                                if str_date.split("-")[-1] == "00":
                                    str_date = str_date.replace("00", "01")
                                if str_date.split("-")[-2] == "00":
                                    str_date = str_date.replace("00", "01")
                                local_date = dt.datetime.strptime(str_date, '%Y-%m-%d').date()
                                if global_date > local_date:
                                    global_date = local_date

                        most_common_sequence = max(most_common_seq_counter)
                        lenght_of_sequence = len(str(most_common_sequence))
                        lenght_of_sequence_no_makaf = len(str(most_common_sequence).replace("-", ""))

                        sequences_meta_data.append(temp_sequence)
                        clade = sequences_meta_data[0].split(" ")[5]
                        variant = sequences_meta_data[0].split(" ")[6]
                        fasta_kind = filename.split('_')[1]

                        # extract ones
                        ones_counter.update(Counter(most_common_sequence))

                        # extract doubles
                        doubels = self.extract_doubles(most_common_sequence)
                        doubles_counter.update(Counter(doubels))

                        # Extract codons
                        codons = self.extract_codons(most_common_sequence)
                        codons_counter.update(Counter(codons))

                        bicodon = self.extract_bio_codons(most_common_sequence)
                        bi_codon_counter.update(bicodon)

                        total_ones_in_gene = self.get_sum_of_dict_values(ones_counter)
                        total_doubles_in_gene = self.get_sum_of_dict_values(doubles_counter)
                        total_codons_in_gene = self.get_sum_of_dict_values(codons_counter)
                        total_bi_codons_in_gene = self.get_sum_of_dict_values(bi_codon_counter)

                        no_makafs_ones_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(ones_counter,
                                                                                                         '-')
                        ones_updated_counter = self.normalized_counter_samples(no_makafs_ones_counter,
                                                                               int(total_ones_in_gene - samples_dropped))

                        no_makafs_doubles_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                            doubles_counter, '-')
                        doubles_updated_counter = self.normalized_counter_samples(no_makafs_doubles_counter,
                                                                                  int(total_doubles_in_gene - samples_dropped))

                        no_makafs_codons_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                            codons_counter, '-')
                        codons_updated_counter = self.normalized_counter_samples(no_makafs_codons_counter,
                                                                                 int(total_codons_in_gene - samples_dropped))

                        no_makafs_bi_codons_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                            bi_codon_counter, '-')
                        bi_codons_updated_counter = self.normalized_counter_samples(no_makafs_bi_codons_counter,
                                                                                    int(total_bi_codons_in_gene - samples_dropped))

                        ones_updated_counter.update(doubles_updated_counter)
                        ones_updated_counter.update(codons_updated_counter)
                        ones_updated_counter.update(bi_codons_updated_counter)

                        all_codons_dict = ones_updated_counter
                        all_codons_dict['fasta_kind'] = fasta_kind
                        all_codons_dict['gene_name'] = directory_name
                        all_codons_dict['sequences_lenght'] = lenght_of_sequence
                        all_codons_dict['sequences_lenght_no_makaf'] = lenght_of_sequence_no_makaf
                        all_codons_dict['variant'] = variant
                        all_codons_dict['clade'] = clade

                    if (len(self.main_df) == 0):
                        self.main_df = pd.DataFrame([all_codons_dict])
                    else:

                        self.main_df = pd.concat([self.main_df, pd.DataFrame([all_codons_dict])], ignore_index=True)
        print(f"Finish extract {directory}")


    def main_extract_fasta_from_dir_(self, directory):
        # iterate over all directories
        for directory_name in tqdm(os.listdir(directory)):
            print(f"\nStart extract {directory_name}")
            # iterate over all fasta files in directory
            if len(os.listdir(directory + '/{}'.format(directory_name))) < 11:
                print(f"dir {directory_name} containe LESS then 11 fastas!!!")
            for filename in os.listdir(directory + '/{}'.format(directory_name)):
                if filename.endswith('.fasta'):
                    # Read the sequences from the FASTA file
                    sequence_counter = 0
                    for seq_index, record in enumerate(
                            SeqIO.parse(directory + directory_name + '/' + filename, "fasta")):
                        # if seq_index == 0:
                        meta_data = str(record.description).split(" ")
                        codons_counter = Counter()
                        doubles_counter = Counter()
                        ones_counter = Counter()
                        sequence_counter += 1

                        # extract ones
                        ones_counter.update(Counter(str(record.seq)))

                        # extract doubles
                        doubels = self.extract_doubles(str(record.seq))
                        doubles_counter.update(Counter(doubels))

                        # Extract codons
                        codons = self.extract_codons(str(record.seq))
                        codons_counter.update(Counter(codons))

                        total_ones_in_fasta = len(record.seq)
                        total_doubles_in_fasta = len(doubels)
                        total_codons_in_fasta = len(codons)

                        if self.counting_makafs:
                            no_makafs_ones_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                                ones_counter, '-')
                            ones_updated_counter = self.normalized_counter_samples(no_makafs_ones_counter,
                                                                                   int(total_ones_in_fasta - samples_dropped))

                            no_makafs_doubles_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                                doubles_counter,
                                '-')
                            doubles_updated_counter = self.normalized_counter_samples(no_makafs_doubles_counter, int(
                                total_doubles_in_fasta - samples_dropped))

                            no_makafs_codons_counter, samples_dropped = self.remove_makaf_from_counter_or_dict(
                                codons_counter,
                                '-')
                            codons_updated_counter = self.normalized_counter_samples(no_makafs_codons_counter,
                                                                                     int(total_codons_in_fasta - samples_dropped))
                        else:
                            ones_updated_counter = self.normalized_counter_samples(ones_counter, total_ones_in_fasta)
                            doubles_updated_counter = self.normalized_counter_samples(doubles_counter,
                                                                                      total_doubles_in_fasta)
                            codons_updated_counter = self.normalized_counter_samples(codons_counter,
                                                                                     total_codons_in_fasta)
                        all_codons_dict = dict(codons_updated_counter + doubles_updated_counter + ones_updated_counter)

                        # add_meta_data_to_dict
                        gene_name, fasta_kind = self.get_meta_data_from_file(filename)
                        all_codons_dict['gene_name'] = gene_name
                        all_codons_dict['fasta_kind'] = fasta_kind
                        all_codons_dict['ID'] = meta_data[0]
                        all_codons_dict['date'] = meta_data[1]
                        all_codons_dict['country'] = meta_data[2]
                        all_codons_dict['continent'] = meta_data[3]
                        all_codons_dict['?'] = meta_data[5]
                        all_codons_dict['variant'] = meta_data[-1]

                        if (len(self.main_df) == 0):
                            self.main_df = self.main_df.from_dict([all_codons_dict])
                        else:
                            self.main_df.loc[len(self.main_df)] = pd.Series(all_codons_dict)
            print(f"Finish extract {directory_name}")


    def main_extract_fasta_from_dir_raw_seq_and_metadata(self, directory):
        # iterate over all directories

        for directory_name in tqdm(os.listdir(directory)):
            sequences_meta_data = []

            if directory_name.endswith('.csv'):
                continue
            # iterate over all fasta files in directory
            if directory_name.startswith('.') == False:
                for filename in tqdm(os.listdir(directory + '{}'.format(directory_name))):
                    if filename.endswith('.fasta'):
                        # Read the sequences from the FASTA file
                        most_common_seq_counter = Counter()
                        temp_sequence = None
                        for index, sequence in enumerate(
                                SeqIO.parse(directory + directory_name + '/' + filename, "fasta")):

                            all_seq_seq = self.extract_all_seq(sequence.seq)
                            most_common_seq_counter.update(all_seq_seq)

                            if index == 0:
                                temp_sequence = sequence.description

                        most_common_sequence = max(most_common_seq_counter)
                        sequences_meta_data.append(temp_sequence)
                        clade = sequences_meta_data[0].split(" ")[5]
                        variant = sequences_meta_data[0].split(" ")[6]
                        fasta_kind = filename.split('_')[1]

                        all_codons_dict = {}
                        all_codons_dict['fasta_kind'] = fasta_kind
                        all_codons_dict['gene_name'] = directory_name
                        all_codons_dict['variant'] = variant
                        all_codons_dict['clade'] = clade
                        all_codons_dict['raw_seq'] = most_common_sequence

                    if (len(self.main_df) == 0):
                        self.main_df = pd.DataFrame([all_codons_dict])
                    else:
                        self.main_df = pd.concat([self.main_df, pd.DataFrame([all_codons_dict])], ignore_index=True)
        print(f"Finish extract {directory}")
