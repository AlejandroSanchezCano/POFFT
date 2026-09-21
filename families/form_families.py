"""
===============================================================================
Title:      Form families
Outline:    Families are formed by clustering the proteins based on their Pfam
            domain architectures. For that, we first keep the significant Pfam
            hits from the hmmscan results, and perform a overlap-aware mapping
            from each protein to its domain architecture. Then, a graph is
            constructed where every protein is a node and proteins are
            connected if they share at least one Pfam domain in the same order
            in their architectures (i.e., they are "equivalent"). Graph
            construction is done by doing pairwise comparisons of the domain
            architectures of proteins that share at least one Pfam domain (this
            is much faster than doing pairwise comparisons of all proteins).
            Finally, the graph is clustered using Markov Clustering (MCL) and a
            representative domain architecture is assigned to each cluster. MCL
            can be tuned with the hyperparameter "inflation"=2, which controls
            the granularity of the clusters. The default value was ultimately
            chosen, but it can be optimized by maximing the modularity metric Q
Docs:       https://github.com/GuyAllard/markov_clustering/tree/master
Author:     Alejandro Sánchez Cano
Date:       17/09/2026
Time:       7 min
===============================================================================
"""

# Built-in modules
import json
import itertools
from collections import defaultdict

# Third-party modules
import scipy
import pandas as pd
from tqdm import tqdm
import networkx as nx
import markov_clustering as mcl

# Custom modules
from misc import paths
from misc import config
from misc.logger import logger
from domtblout import Domtblout
logger.info('Importing modules completed')

###############################################################################
#######                       CONSTRUCT NETWORK                         #######
###############################################################################

# Load domtblout
domtblout_file = paths.FAMILIES / 'hmmscan.domtblout'
domtblout = Domtblout.from_df_file(domtblout_file)

# Filter by i_evalue threshold
domtblout.evalue_filter(
    field='i_evalue', 
    threshold=config.IEVALUE_THRESHOLD
)
logger.info(
    f'Filtered Pfam hits by i_evalue threshold of {config.IEVALUE_THRESHOLD}, '
    f'keeping {len(domtblout.df)} hits and '
    f'{len(domtblout.df["query_name"].unique())} unique queries'
)

# Accession-to-architecture mapping
accession2architecture, _ = domtblout.solve_overlap(config.OVERLAP_THRESHOLD)

# Since we have previously filtered interactions based on whether both proteins
# had hmmscan hits, we should construct the graph based on the accessions from
# the filtered set of proteins. However, in our case, there number of queries
# with hmmscan hits (76,083) is the same as the number of accessions in the
# domtblout file (76,083). This means that no additional protein has been 
# affected by their partner being filtered out. Therefore, we can proceed.
# Else, it would not matter much because the upcoming analysis will be made on
# interaction and sequence files, which are properly filtered. The only thing
# affected is the graph construction, which would give a graph with more nodes
# and edges, but that is not a problem. This, this is just a disclaimer.

# Reverse mapping: PFAM to accessions
pfam2accessions = defaultdict(set) # avoids self-loops in the graph
for accession, architecture in accession2architecture.items():
    for domains in architecture:
        for domain in domains:
            pfam2accessions[domain].add(accession)
logger.info(
    f'Mapped {len(pfam2accessions)} unique Pfam domains to '
    f'{len(accession2architecture)} unique accessions'
)

# Construct graph
graph = nx.Graph()
graph.add_nodes_from(accession2architecture.keys())
for pfam, accessions in tqdm(pfam2accessions.items(), desc='Adding edges'):
    for accession1, accession2 in itertools.combinations(accessions, 2):
        
        # Get architectures for both queries
        architecture1 = accession2architecture[accession1]
        architecture2 = accession2architecture[accession2]

        # Check if architectures are equivalent
        if architecture1.equivalent(architecture2):
            graph.add_edge(accession1, accession2)

logger.info(
    f'Constructed graph with {graph.number_of_nodes()} nodes and '
    f'{graph.number_of_edges()} edges'
)

###############################################################################
#######                       MARKOV CLUSTERING                         #######
###############################################################################

# Run MCL clustering
matrix = nx.to_scipy_sparse_array(graph).asformat("csr")
matrix = scipy.sparse.csr_matrix(matrix)
result = mcl.run_mcl(matrix, inflation=1.1)           
clusters = mcl.get_clusters(result)
logger.info(f'Found {len(clusters)} clusters using MCL')

# Find representative architecture for each cluster
clusterindex2architecture = {}
nodes = list(graph.nodes())
for idx, cluster in enumerate(clusters):
    counts = defaultdict(int)
    for jdx in cluster:
        architecture = accession2architecture[nodes[jdx]]
        counts[architecture] += 1
    representative_architecture = max(counts, key=counts.get)
    clusterindex2architecture[idx] = representative_architecture

# Map accession to cluster representative architecture
accession2representative = {}
for idx, cluster in enumerate(clusters):
    representative_architecture = clusterindex2architecture[idx]
    for jdx in cluster:
        accession = nodes[jdx]
        representative = representative_architecture.architecture # json serializable
        accession2representative[accession] = representative

# Save mapping
with open(paths.FAMILIES / 'accession2representative.json', 'w') as handle:
    json.dump(accession2representative, handle, indent=4)