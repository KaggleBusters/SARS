import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway


class Sars_Analyzer:

    def __init__(self, data):
        self.data = data

    def WorlByVariant(self):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world = world.merge(self.data, left_on='name', right_on='country', how='left')
        fig, ax = plt.subplots(1, 1)
        world.plot(column='variant', ax=ax, legend=True, cmap='YlOrRd')
        plt.show()

    def features_correlation_heatmap(self, type):
        '''
        :param type: can be 3, 2, or 1 (3 for codons, 2 for dual nucleotides, and 1 for single nucleotides)
        :return: plots the correlation heatmap
        '''



        column_names = self.data.columns[self.data.columns.str.len() == type].tolist()
        corr_matrix = self.data[column_names].corr()
        plt.figure(figsize=(10, 8))  # Set the figure size
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title(f'Correlation Heatmap for consecutive {type} nucleotides')
        plt.show()

    def scatterplot_dna_percentage_by_continent_and_date(self, dna_percentage, continent, date):
        '''
        Creates a scatter plot to visualize the dna_percentage % content compared to country and date.
         # in GC for example it helps to find what continent and at what date had the highest heat resistence (GC content)
        :param dna_percentage: Column name for dna percentage content
        :param continent: Column name for continent
        :param date: Column name for date

        The x-axis represents the date, the y-axis represents the GC content, and the points are colored based on the country.
        '''

        # Select the columns for GC content, country, and date from the data
        gc_column = self.data[dna_percentage]
        continent_column = self.data[continent]
        date_column = self.data[date].sort_values(ascending=True)

        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=date_column, y=gc_column, hue=continent_column, palette='Set1')
        plt.xlabel('Date')
        plt.ylabel('GC Content')
        plt.title('GC Content Compared to continent and Date')
        plt.legend(title='Country', bbox_to_anchor=(1, 1))
        plt.show()

    def evolutionary_path_for_pango_lineage(self, lineage):
        # displays the provided pango lineage by printing the evolutionary paths for each gene
        lineage_analysis = self.trace_taxonomy_dictionary(lineage)

        # Print the evolutionary paths for the pango lineage
        for gene, paths in lineage_analysis.items():
            print(f"Gene: {gene}")
            for path in paths:
                print(" -> ".join(map(str, path)))
            print()

    def network_plot_gene(self, lineage):
        # Create a network plot to visualize the evolutionary connections between genes in the pango lineage
        network_plot_digraph = nx.DiGraph()
        lineage_analysis = self.trace_taxonomy_dictionary(lineage)
        for gene, paths in lineage_analysis.items():
            network_plot_digraph.add_node(gene)
            for path in paths:
                for i in range(len(path) - 1):
                    network_plot_digraph.add_edge(path[i], path[i + 1])

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(network_plot_digraph)
        nx.draw_networkx(network_plot_digraph, pos, with_labels=True, node_color='lightblue', edge_color='gray',
                         arrowsize=12)
        plt.title('Evolutionary Connections in Pango Lineage')
        plt.show()

    def stacked_barplot_pango_lineage(self, lineage, column):
        '''
        Plots a stacked bar chart to compare the frequency of a specific column for each gene in the Pango lineage.

        :param lineage: Pango lineage to analyze
        :param column: Column name to consider for frequency analysis
        '''

        lineage_analysis = self.trace_taxonomy_dictionary(lineage)

        # Get the values for the specified column for each gene in the Pango lineage
        gene_values = {}
        for gene, paths in lineage_analysis.items():
            values = [self.data.loc[(self.data['rank 1'] == gene) & (self.data['rank 2'] == path[0]) &
                                    (self.data['rank 3'] == path[1]) & (self.data['rank 4'] == path[2]),
            column].values[0] for path in paths]
            gene_values[gene] = values

        # Calculate the frequency of each value in the specified column
        unique_values = np.unique(self.data[column].values)
        frequencies = {gene: [values.count(value) for value in unique_values] for gene, values in gene_values.items()}

        # Create a stacked bar chart for the frequency of the specified column for each gene
        plt.figure(figsize=(8, 6))
        bottom = np.zeros(len(unique_values))
        for gene, gene_frequencies in frequencies.items():
            plt.bar(unique_values, gene_frequencies, bottom=bottom, label=gene)
            bottom += gene_frequencies
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Frequency of {column} in Pango Lineage')
        plt.legend()
        plt.show()

    def histogram_pango_lineage(self, lineage, column):
        '''
        Plots a histogram to visualize the distribution of a specific column for each gene in the pango lineage.

        :param lineage: pango lineage to analyze
        :param column: column name to consider for distribution analysis
        '''

        lineage_analysis = self.trace_taxonomy_dictionary(lineage)

        # Get the values for the specified column for each gene in the pango lineage
        gene_values = {}
        for gene, paths in lineage_analysis.items():
            values = [self.data.loc[(self.data['rank 1'] == gene) & (self.data['rank 2'] == path[0]) &
                                    (self.data['rank 3'] == path[1]) & (self.data['rank 4'] == path[2]),
            column].values[0] for path in paths]
            gene_values[gene] = values

        # Plot histogram for the specified column's distribution for each gene
        plt.figure(figsize=(8, 6))
        for gene, values in gene_values.items():
            sns.histplot(values, label=gene, kde=True)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'Distribution of {column} in Pango Lineage')
        plt.legend()
        plt.show()

    def highlight_highest_dna_sequence_values(self, type):
        '''
        Creates a bar plot to highlight the highest value of each DNA sequence column.

        param type: can be 3, 2, or 1 (3 for codons, 2 for dual nucleotides, and 1 for single nucleotides)
        :return: plots the correlation heatmap
        '''

        dna_columns = self.data.columns[self.data.columns.str.len() == type].tolist()
        # Select the DNA sequence columns
        # dna_columns = self.data.columns[:58]  # Adjust the index range based on your specific data

        # Convert the data in DNA sequence columns to numeric values
        self.data[dna_columns] = self.data[dna_columns].apply(pd.to_numeric, errors='coerce')

        # Remove any non-numeric values or handle them according to your requirements
        self.data[dna_columns] = self.data[dna_columns].fillna(0)

        # Calculate the highest value for each DNA sequence column
        highest_values = self.data[dna_columns].max() * 100

        # Create a bar plot to visualize the highest values
        plt.figure(figsize=(10, 8))
        highest_values.plot(kind='bar', color='blue', alpha=0.7, label='Highest Value')

        plt.xlabel('DNA Sequence')
        plt.ylabel('Value as (%)')
        plt.title('Highest Value of DNA Sequences')
        plt.legend()
        plt.show()

    def scatterplot_dna_percentage_correlation(self, negative_1, negative_2):
        '''
        Creates a scatter plot to visualize the negative correlation between 'AT' and 'TG' columns for example.
        '''

        # Select the 'AT' and 'TG' columns from the data - example
        at_column = self.data[negative_1]
        tg_column = self.data[negative_2]

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=at_column, y=tg_column)
        plt.xlabel(f'{negative_1}')
        plt.ylabel(f'{negative_2}')
        plt.title(f'Scatter Plot: Correlation between {negative_1} and {negative_2}')
        plt.show()

    def traceTaxonomy(self, gene):

        '''
        Helper function which will return the evolution path in each gene
        :param gene: gene name (A,AA,XBC, etc...)
        :return: list of lists with the evolutionary paths
        '''

        d = self.data.copy()
        d1 = d[d['rank 1'] == gene]
        paths = d1.groupby(['rank 1', 'rank 2', 'rank 3', 'rank 4']).count().index

        return list(paths)

    def trace_taxonomy_dictionary(self, gene):
        '''
        Helper function which will return the evolution path in each gene
        :param gene: gene name (A, AA, XBC, etc...)
        :return: dictionary with the evolutionary paths
        '''

        d = self.data.copy()
        d1 = d[d['rank 1'] == gene]
        paths = d1.groupby(['rank 1', 'rank 2', 'rank 3', 'rank 4']).count().index

        lineage_dict = {}

        for path in paths:
            key = gene
            values = list(path[1:])  # Convert the values to a list
            if key in lineage_dict:
                lineage_dict[key].append(tuple(values))
            else:
                lineage_dict[key] = [tuple(values)]

        return lineage_dict

    def trace_taxonomy_without_zeroes(self, gene):
        '''
        Helper function which will return the evolution path in each gene
        :param gene: gene name (A, AA, XBC, etc...)
        :return: dictionary with the evolutionary paths
        '''

        d = self.data.copy()
        d1 = d[d['rank 1'] == gene]
        paths = d1.groupby(['rank 1', 'rank 2', 'rank 3', 'rank 4']).count().index

        lineage_dict = {}

        for path in paths:
            key = gene
            values = [v if v != 0 else '' for v in path[1:]]  # Remove zeroes from the tuple
            if key in lineage_dict:
                lineage_dict[key].append(tuple(values))
            else:
                lineage_dict[key] = [tuple(values)]

        return lineage_dict

    def featureEvolution(self, type, gene, extraction_type):
     '''
     :param type: can be 3,2 or 1. (3 for codons, 2 for dual nucleotides and 1 for single nucleotides)
     :param gene: gene name (A,AA,XBC, etc...)
     :param extraction_type: E or Spike, etc..
     :return: plots the features mean value through the evolution
     '''

     d = self.data.copy()
     d = d[d['fasta_kind'] == extraction_type]
     relevant_evolutionary_paths = self.traceTaxonomy(gene)
     column_names = d.columns[d.columns.str.len() == type].tolist() + ['rank 1','rank 2','rank 3','rank 4']
     filtered_data = d[column_names].copy()

     num_cols = filtered_data.shape[1] - 5
     split_lists = [[] for i in range(num_cols)]

     for path in relevant_evolutionary_paths:
         d = filtered_data.copy()
         first_filter = d['rank 1'] == path[0]
         second_filter = d['rank 2'] == path[1]
         third_filter = d['rank 3'] == path[2]
         fourth_filter = d['rank 4'] == path[3]
         d = d[first_filter & second_filter & third_filter & fourth_filter]

         avg_list = d[d.columns[:num_cols]].mean(axis = 0)
         split_lists_2 = [avg_list[i:i + 1] for i in range(num_cols)]

         for i, lst in enumerate(split_lists_2):
             split_lists[i].append(list(lst)[0])

     plt.figure(figsize=(15, 8))
     labels = list(d.columns[:num_cols])
     x = [str(i[0]) + str(i[1]) + str(i[2]) + str(i[3]) for i in list(relevant_evolutionary_paths)]
     for i, y in enumerate(split_lists):
         plt.plot(x, list(y), label = labels[i])

     plt.xlabel('gene')
     plt.ylabel('feature mean')
     plt.title(f'{type} consecutive nucleotides features evolution for gene {gene} with fasta kind {extraction_type}')

     plt.xticks(x)
     plt.xticks(rotation=90)
     plt.legend()

     plt.show()

    def start_fix(self):
        self.data.pop("Unnamed: 0")
        self.data[['rank 1', 'rank 2', 'rank 3', 'rank 4']] = self.data['gene_name'].str.split(".", expand=True)
        self.data.fillna(0, inplace=True)
        self.data['rank 2'] = self.data['rank 2'].replace('', 0)
        self.data['rank 3'] = self.data['rank 3'].replace('', 0)
        self.data['rank 4'] = self.data['rank 4'].replace('', 0)
        self.data.rename(columns={'?': 'clade', 'ID': 'gisaid_id', 'fasta_kind': 'gene', 'gene_name': 'pango_lineage'})

    def perform_density_plot(self, type_col, category1, category2=None, fill=True):
        '''
            The perform_density_plot function is a method of the analyzer class that displays a kernel density estimate
            of the distribution of features for one or two categories in a dataframe. The function takes four arguments:

            type_col: The length of the column names to use for the plot. The function will select columns from the dataframe whose names have this length.

            category1: The first category to plot. The function will display the distribution of the features for this category.

            category2: An optional second category to plot. If specified, the function will display the distribution of the features for this category as well.

            fill: An optional boolean argument that specifies whether to fill the area under the distribution curve. If set to True, the function will fill the area under the curve with a semi-transparent color.
        '''

        columns = self.data.columns[self.data.columns.str.len() == type_col].tolist()[:-1]
        plt.figure(figsize=(12, 8))
        if category2:
            data1 = self.data[self.data['rank 1'] == category1][columns].copy()
            data2 = self.data[self.data['rank 1'] == category2][columns].copy()
            for col in columns:
                density1 = data1[col].plot.kde(label=col)
                density2 = data2[col].plot.kde(label=col)
                if fill:
                    x1, y1 = density1.lines[-1].get_data()
                    x2, y2 = density2.lines[-1].get_data()
                    plt.fill_between(x1, y1, alpha=0.3)
                    plt.fill_between(x2, y2, alpha=0.4)
        else:
            data = self.data[self.data['rank 1'] == category1][columns].copy()
            for col in columns:
                density = data[col].plot.kde(label=col)
                if fill:
                    x, y = density.lines[-1].get_data()
                    plt.fill_between(x, y, alpha=0.3)

        plt.legend()
        if category2:
            plt.title(f'Distribution of {category1} and {category2}')
        else:
            plt.title(f'Distribution of {category1}')
        plt.show()

    def perform_strain_analysis(self, strains_column, sequence_columns, strain_group1, strain_group2):

        if isinstance(sequence_columns, int):
            sequence_columns = self.data.columns[self.data.columns.str.len() == sequence_columns].tolist()[:-1]

        group1_data = self.data[self.data[strains_column].isin(strain_group1)]
        group2_data = self.data[self.data[strains_column].isin(strain_group2)]

        for sequence_column in sequence_columns:
            _, p_value = ttest_ind(group1_data[sequence_column], group2_data[sequence_column])
            print(f"T-test p-value for {sequence_column}: {p_value}")

            _, p_value = f_oneway(group1_data[sequence_column], group2_data[sequence_column])
            print(f"One-Way ANOVA p-value for {sequence_column}: {p_value}")

            if p_value < 0.05:
                reject, corrected_p_values, _, _ = multipletests([p_value], method='bonferroni')
                print(f"Corrected p-values for {sequence_column}: {corrected_p_values}")

    def perform_AB_population_test_for_comb(self, formula_column, start_quadrant_column, dependent_columns):
        formula_categories = self.data[formula_column].unique()
        start_quadrant_categories = self.data[start_quadrant_column].unique()
        dependent_columns = self.data.columns[self.data.columns.str.len() == dependent_columns].tolist()[:-1]

        for formula_category in formula_categories:
            for start_quadrant_category in start_quadrant_categories:
                for dependent_column in dependent_columns:
                    # Population A: Specific formula category and start_quadrant category
                    data_A = self.data[(self.data[formula_column] == formula_category) & (
                                self.data[start_quadrant_column] == start_quadrant_category)][dependent_column]

                    # Population B: Rest of the categories
                    rest_formula_categories = [category for category in formula_categories if category != formula_category]
                    rest_start_quadrant_categories = [category for category in start_quadrant_categories if
                                                      category != start_quadrant_category]
                    data_B = self.data[(self.data[formula_column].isin(rest_formula_categories)) & (
                        self.data[start_quadrant_column].isin(rest_start_quadrant_categories))][dependent_column]

                    # Check if populations have non-zero size
                    if len(data_A) == 0 or len(data_B) == 0:
                        print(
                            f"Comparison: {formula_category} - {start_quadrant_column} {start_quadrant_category} vs. Rest")
                        print("One of the populations has zero size. Skipping the tests for this combination.\n")
                        continue

                    # Perform t-test for population A vs. population B
                    t_statistic, p_value_t = ttest_ind(data_A, data_B, equal_var=False)

                    # Perform Wilcoxon test for population A vs. population B
                    statistic, p_value_w = mannwhitneyu(data_A, data_B, alternative='two-sided')

                    # Adjust p-values using Bonferroni correction
                    alpha = 0.05 / (len(formula_categories) * len(start_quadrant_categories))
                    p_value_t_adjusted = p_value_t * (len(formula_categories) * len(start_quadrant_categories))
                    p_value_w_adjusted = p_value_w * (len(formula_categories) * len(start_quadrant_categories))

                    counter = 0
                    # Print results for the specific combination
                    print(f"Comparison: {formula_category} - {start_quadrant_column} {start_quadrant_category} vs. Rest for {dependent_column}")
                    print(f"t-Test: p-value = {p_value_t:.4f}")
                    if p_value_t_adjusted < alpha:
                        print("The difference is statistically significant (t-test)!!!!!!!!!!!!!!")
                        counter += 1

                    print(f"Mann-Whitney test: p-value = {p_value_w:.4f}")
                    if p_value_w_adjusted < alpha:
                        print("The difference is statistically significant (Mann-Whitney test)!!!!!!!!!!!!!!")
                        counter += 1

                    # Perform permutation test for category vs. rest
                    max_perm = int(1e4)  # Maximum number of permutations
                    perm_diffs = []
                    np.random.seed(42)  # Set random seed for reproducibility

                    for _ in range(max_perm):
                        perm_data = np.concatenate((data_A, data_B))
                        np.random.shuffle(perm_data)
                        perm_diff = np.mean(perm_data[:len(data_A)]) - np.mean(perm_data[len(data_A):])
                        perm_diffs.append(perm_diff)

                    perm_p_value = np.mean(np.abs(perm_diffs) >= np.abs(np.mean(data_A) - np.mean(data_B)))
                    p_value_perm_adjusted = perm_p_value * (len(formula_categories) * len(start_quadrant_categories))
                    print(f"Permutation test: p-value = {perm_p_value:.10f} \n")

                    if p_value_perm_adjusted < alpha:
                        print("The difference is statistically significant (permutation test)!!!!!!!!!!!!!!")
                        counter += 1
                        print("--------------------------------------------\n")

                    if counter == 3:
                        # Calculate and display the average and standard deviation
                        avg_A = np.mean(data_A)
                        std_A = np.std(data_A)
                        avg_B = np.mean(data_B)
                        std_B = np.std(data_B)

                        print(f"\nAverage of Population A: {avg_A:.4f} +- {std_A:.4f}")
                        print(f"Average of Population B: {avg_B:.4f} +- {std_B:.4f}")

                        # Visualize the distributions
                        plt.figure(figsize=(8, 6))
                        plt.hist(data_A, bins=10, alpha=0.5, label='Population A')
                        plt.hist(data_B, bins=10, alpha=0.5, label='Population B')
                        plt.xlabel(dependent_column)
                        plt.ylabel('Frequency')
                        plt.title(f"Distributions: {formula_category} + {start_quadrant_category} vs. Rest")
                        plt.legend()
                        plt.show()
                        print('\n\n\n')

    def perform_AB_population_test_with_plot(self, formula_column, dependent_columns):

        # Get unique categories from the formula column
        categories = self.data[formula_column].unique()
        dependent_columns = self.data.columns[self.data.columns.str.len() == dependent_columns].tolist()[:-1]

        # Set the significance level
        alpha = 0.05
        adjusted_alpha = alpha / len(categories)

        # Perform A/B population tests for each category
        for category in categories:
            for dependent_column in dependent_columns:
                rest_categories = [c for c in categories if c != category]
                data_A = self.data[self.data[formula_column] == category][dependent_column]
                data_B = self.data[self.data[formula_column].isin(rest_categories)][dependent_column]

                print(f"Comparison: {category} vs. Rest ({', '.join(rest_categories)})")

                counter = 0
                # Perform t-test for category vs. rest
                t_statistic, p_value_t = ttest_ind(data_A, data_B)
                p_value_t_adjusted = p_value_t * len(categories)
                print(f"t-Test: p-value = {p_value_t:.10f}")

                if p_value_t_adjusted < adjusted_alpha:
                    print("The difference is statistically significant (t-test)")
                    counter += 1
                else:
                    print("The difference is NOT statistically significant (t-test)!!!")

                # Perform permutation test for category vs. rest
                max_perm = int(1e4)  # Maximum number of permutations
                perm_diffs = []
                np.random.seed(42)  # Set random seed for reproducibility

                for _ in range(max_perm):
                    perm_data = np.concatenate((data_A, data_B))
                    np.random.shuffle(perm_data)
                    perm_diff = np.mean(perm_data[:len(data_A)]) - np.mean(perm_data[len(data_A):])
                    perm_diffs.append(perm_diff)

                perm_p_value = np.mean(np.abs(perm_diffs) >= np.abs(np.mean(data_A) - np.mean(data_B)))
                p_value_perm_adjusted = perm_p_value * len(categories)
                print(f"Permutation test: p-value = {perm_p_value:.10f}")

                if p_value_perm_adjusted < adjusted_alpha:
                    print("The difference is statistically significant (permutation test)")
                    counter += 1
                else:
                    print("The difference is NOT statistically significant (permutation test)!!!")

                if counter == 2:
                    # Calculate and display the average and standard deviation
                    avg_A = np.mean(data_A)
                    std_A = np.std(data_A)
                    avg_B = np.mean(data_B)
                    std_B = np.std(data_B)

                    print(f"\nAverage of Population A: {avg_A:.4f} +- {std_A:.4f}")
                    print(f"Average of Population B: {avg_B:.4f} +- {std_B:.4f}")

                    # Visualize the distributions
                    plt.figure(figsize=(8, 6))
                    plt.hist(data_A, bins=20, alpha=0.5, label='Population A')
                    plt.hist(data_B, bins=20, alpha=0.5, label='Population B')
                    plt.xlabel(dependent_column)
                    plt.ylabel('Frequency')
                    plt.title(f"Distributions: {category} vs. Rest")
                    plt.legend()
                    plt.show()

                    print('\n\n\n')

    def featureHistogramPerGene(self, feature, type):
        '''
        :param feature: name of the feature
        :param type: can be 3,2 or 1. (3 for codons, 2 for dual nucleotides and 1 for single nucleotides)
        :return: plots the histograms for
        '''

        pass

    def generate_color_fades(self,n):

        start_color = '#FB0000'
        end_color = '#FEC9C9'

        # Convert start and end colors to RGB values
        start_rgb = tuple(int(start_color[i:i + 2], 16) for i in (1, 3, 5))
        end_rgb = tuple(int(end_color[i:i + 2], 16) for i in (1, 3, 5))

        # Calculate the color increment for each step
        increment = tuple((end_rgb[i] - start_rgb[i]) / (n - 1) for i in range(3))

        # Generate the list of color codes
        color_fades = [start_color]
        for i in range(1, n):
            r = int(start_rgb[0] + i * increment[0])
            g = int(start_rgb[1] + i * increment[1])
            b = int(start_rgb[2] + i * increment[2])
            color_code = "#{:02x}{:02x}{:02x}".format(r, g, b)
            color_fades.append(color_code)

        return color_fades

    def geneTravelPathWorldWild(self, gene):

        d = self.data[self.data['rank 1'] == gene].copy()
        relevant_evolutionary_paths = self.traceTaxonomy(gene)
        faded_colors = self.generate_color_fades(len(relevant_evolutionary_paths))

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        ax = world.plot()

        for i,path in enumerate(relevant_evolutionary_paths):
            d1 = d.copy()
            first_filter = d['rank 1'] == path[0]
            second_filter = d['rank 2'] == path[1]
            third_filter = d['rank 3'] == path[2]
            fourth_filter = d['rank 4'] == path[3]
            d1 = d1[first_filter & second_filter & third_filter & fourth_filter]

            world[world.name == d1['country'].iloc[0]].plot(color=faded_colors[i], ax=ax)

        plt.title(f"Evolutionary path for gene {gene}")
        plt.show()

    def seq_len_count_country_by_gen(self, gene, x_col, plot_or_save, country_flag=True):
        gene_list = self.traceTaxonomy(gene)

        # Concatenate data for all the genes with 'gene' in their name
        gene_dfs = [self.data[
                        (self.data['rank 1'] == gene_info[0]) &
                        (self.data['rank 2'] == gene_info[1]) &
                        (self.data['rank 3'] == gene_info[2]) &
                        (self.data['rank 4'] == gene_info[3])
                        ] for gene_info in gene_list]
        merged_data = pd.concat(gene_dfs)

        self.plot_violin_lengths(merged_data, gene, x_col, plot_or_save)
        self.plot_violin_total_sequences(merged_data, gene, x_col, plot_or_save)
        if country_flag:
            self.plot_violin_country_reports(merged_data, gene, plot_or_save)  # If needed

    def plot_violin_lengths(self, gene_data, gene, x_col, plot_or_save):

        plt.figure(figsize=(10, 6))
        plt.title(f"Lengths of Sequences for gen {gene}")
        plt.xlabel("Gene Variety")
        plt.ylabel("Sequences Length")
        sns.violinplot(x=x_col, y='sequences_lenght', hue=x_col, data=gene_data, inner="quart")
        plt.xticks(rotation=45, ha='right')
        plt.legend()

        if plot_or_save == 'plot':
            plt.show()
        else:
            plt.savefig(plot_or_save+f'/Lengths_of_Sequences_for_gen_{gene}_by_{x_col}')

    def plot_violin_total_sequences(self, gene_data, gene, x_col, plot_or_save):
        # Convert 'total_sequences' column to numeric data type and handle non-numeric values
        gene_data['total_sequences'] = pd.to_numeric(gene_data['total_sequences'], errors='coerce').fillna(0).astype('int64')
        plt.figure(figsize=(10, 6))
        plt.title(f"Amount of Sequences for gene {gene}")
        plt.xlabel(x_col)
        plt.ylabel("Total Sequences")
        sns.violinplot(x=x_col, y='total_sequences', hue=x_col, data=gene_data, inner="quart")
        plt.xticks(rotation=45, ha='right')
        plt.legend()

        if plot_or_save == 'plot':
            plt.show()
        else:
            plt.savefig(plot_or_save + f'/Amount_of_Sequences_for_gen_{gene}_by_{x_col}')

    def extract_country_counts(self, row, countries):
        countries_data = eval(row['countries'])  # Convert the string representation to a dictionary
        return [countries_data.get(country, 0) for country in countries]

    def plot_violin_country_reports(self, merged_data, gene, plot_or_save):
        countries = set()

        # Extract all unique country names from the 'countries' column
        for country_data in merged_data['countries']:
            countries_data = eval(country_data)  # Convert the string representation to a dictionary
            countries.update(countries_data.keys())

        # Extract counts for each country
        merged_data[list(countries)] = merged_data.apply(lambda row: self.extract_country_counts(row, countries),
                                                         axis=1,
                                                         result_type='expand')

        plt.figure(figsize=(10, 8))
        plt.title(f"Number of Reports in Each Country for gen {gene}")
        plt.xlabel("Number of Reports")
        plt.ylabel("Country")
        plt.violinplot(merged_data[list(countries)].values.T, vert=False)
        plt.yticks(range(1, len(countries) + 1), countries, rotation=15)
        plt.legend()

        if plot_or_save == 'plot':
            plt.show()
        else:
            plt.savefig(plot_or_save + f'/Number_of_Reports_in_Each_Country_for_gen_{gene}')

    def feat_plots_by_gen(self, gene, x_col, y_col_type, plot_or_save):

        if type(y_col_type) == int:
            column_names = self.data.columns[self.data.columns.str.len() == y_col_type].tolist()
        else:
            column_names = [y_col_type]
        for y_col in column_names:
            gene_list = self.traceTaxonomy(gene)

            # Concatenate data for all the genes with 'gene' in their name
            gene_dfs = [self.data[
                            (self.data['rank 1'] == gene_info[0]) &
                            (self.data['rank 2'] == gene_info[1]) &
                            (self.data['rank 3'] == gene_info[2]) &
                            (self.data['rank 4'] == gene_info[3])
                            ] for gene_info in gene_list]
            merged_data = pd.concat(gene_dfs)

            self.plot_bar_for_feat(merged_data, gene, x_col, y_col, plot_or_save)

    def plot_bar_for_feat(self, gene_data, gene, x_col, y_col, plot_or_save):

        plt.figure(figsize=(10, 6))
        plt.title(f"normalized frequencies for gen {gene} by {y_col} feature")
        plt.xlabel("Gene Variety")
        plt.ylabel(f"normalized frequencies of {y_col}")
        if x_col == 'pango_lineage':
            sns.barplot(x=x_col, y=y_col, data=gene_data, color='blue')
        else:
            sns.violinplot(x=x_col, y=y_col, data=gene_data, color='blue')

        plt.ylim(np.min(gene_data[y_col])-0.0005, np.max(gene_data[y_col])+0.0005)
        plt.xticks(rotation=45, ha='right')

        if plot_or_save == 'plot':
            plt.show()
        else:
            plt.savefig(plot_or_save+f'/normalized_frequencies_for_gen_{gene}_by_{y_col}_feature_{x_col}')