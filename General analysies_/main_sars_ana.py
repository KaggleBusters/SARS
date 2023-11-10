import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from sars_analyzer import Sars_Analyzer


if __name__ == '__main__':

    sars_data = pd.read_csv('with_entropies_and_bicodons.csv')
    analyzer = Sars_Analyzer(sars_data)
    analyzer.start_fix()

    gene_names_list = analyzer.data['rank 1'].unique().tolist()
    col_names = ['pango_lineage', 'clade', 'variant']

    PLOT_OR_SAVE = r'C:\Users\almog\PycharmProjects\SARS\seq_len_count_country_by_gen'  ## paht to save OR "plot"
    for col_name in col_names:
        for gene_name in gene_names_list:
            analyzer.seq_len_count_country_by_gen(gene_name, col_name, PLOT_OR_SAVE)
            print(f'finished plots {col_name} for {gene_name}')

    PLOT_OR_SAVE = r'C:\Users\almog\PycharmProjects\SARS\entropy_by_gen' # r'C:\Users\almog\PycharmProjects\SARS\feat_plots_by_gen'

    features = ['entropy1', 'entropy2', 'entropy3', 'entropy6']
    for col_name in col_names:
        for gene_name in gene_names_list:
            for feature in features:
                analyzer.feat_plots_by_gen(gene_name, col_name, feature, PLOT_OR_SAVE)
                print(f'finished plots {col_name} for {gene_name} gen and {feature} feat')

    analyzer.WorlByVariant()
    analyzer.geneTravelPathWorldWild('A')
    analyzer.featureEvolution(2, 'B', 'Spike')
    analyzer.features_correlation_heatmap(1) #1/2/3
    analyzer.features_correlation_heatmap(3)
    print(analyzer.trace_taxonomy_dictionary("AM"))
    analyzer.evolutionary_path_for_pango_lineage('XBC')
    analyzer.stacked_barplot_pango_lineage('XBC', 'continent')
    analyzer.network_plot_gene('XBC')
    analyzer.histogram_pango_lineage('C', 'country')
    analyzer.scatterplot_dna_percentage_correlation('AC', 'GA')
    analyzer.scatterplot_dna_percentage_by_continent_and_date('GC', 'continent', 'date')
    analyzer.highlight_highest_dna_sequence_values(1)
    dependent_column = 1
    analyzer.perform_AB_population_test_for_comb('gene_name', 'fasta_kind', dependent_column)
    analyzer.perform_AB_population_test_with_plot('gene_name', dependent_column)
    analyzer.perform_strain_analysis('gene_name', sequence_columns=2,
                                     strain_group1=['A'], strain_group2=['B'])
    analyzer.perform_density_plot(2, 'Z')
    analyzer.featureEvolution(3, 'C')
    analyzer.features_correlation_heatmap(1) #1/2/3


