import time
import pandas as pd
from fasta_handler import Fasta_Handler


def main(name):
    # Path to the main data dir E:\SARS

    directory = "E:\\\SARS\\" # directory = "E:\\SARS_BEQEILO\\"
    sars_df = pd.DataFrame()

    fasta_hand = Fasta_Handler(sars_df)
    start_time = time.time()

    # fasta_hand.main_extract_fasta_from_dir_by_kind(directory)
    fasta_hand.main_extract_fasta_from_dir_raw_seq_and_metadata(directory)
    print(f"Time taken for all fasta extraction in {directory} no entropy:", (time.time() - start_time) / 60)

    fasta_hand.entropy_1()
    fasta_hand.entropy_2()
    fasta_hand.entropy_3()
    fasta_hand.entropy_6()

    data_fk = fasta_hand.main_df['fasta_kind']
    data_gm = fasta_hand.main_df['gene_name']
    data_sl = fasta_hand.main_df['sequences_lenght']
    data_slnm = fasta_hand.main_df['sequences_lenght_no_makaf']
    data_v = fasta_hand.main_df['variant']
    data_c = fasta_hand.main_df['clade']
    fasta_hand.main_df.drop(columns=['fasta_kind', 'gene_name', 'sequences_lenght',
                                     'sequences_lenght_no_makaf', 'variant', 'clade'], inplace=True)

    fasta_hand.main_df['fasta_kind'] = data_fk
    fasta_hand.main_df['gene_name'] = data_gm
    fasta_hand.main_df['sequences_lenght'] = data_sl
    fasta_hand.main_df['sequences_lenght_no_makaf'] = data_slnm
    fasta_hand.main_df['variant'] = data_v
    fasta_hand.main_df['clade'] = data_c

    fasta_hand.main_df = fasta_hand.main_df.fillna(0)
    fasta_hand.main_df.to_csv(directory + 'SARS_only_raw_seq_and_mata_data.csv')
    print(f"Time taken for all fasta extraction in {directory} with entropy:", (time.time() - start_time) / 60)

    df_ = pd.read_csv('E:\SARS\SARS_with_entropy_28K_new_new_hope_new_final.csv')


if __name__ == '__main__':
    main('finished run main')
