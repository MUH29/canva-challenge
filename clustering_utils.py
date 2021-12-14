import itertools
import math
import random
from statistics import mean
from typing import Tuple, List, Dict, Set

# Helper types for the clustering task.
Embedding = Tuple[float, ...]
ClusterId = int
ClusteredDataset = Dict[ClusterId, List[Embedding]]


# Helper functions for the clustering task.
def euclidean_distance(u: Embedding, v: Embedding) -> float:
    """ Returns the Euclidean distance between two Embeddings. """
    return math.sqrt(sum([(uu - vv) ** 2 for uu, vv in zip(u, v)]))


def calculate_mean(embeddings: List[Embedding]) -> Embedding:
    """Returns an embedding where the value is the mean of the embeddings at each dimension.
    In numpy this would be the np.mean(embeddings_matrix, axis=0).
    """
    assert len(embeddings) > 0 and all(
        len(e) == len(embeddings[0]) for e in embeddings[1:]
    ), "Require more than one embedding, all of the same dimension."
    dim = len(embeddings[0])
    return tuple(mean(e[i] for e in embeddings) for i in range(dim))


"""
         888                     
         888                     
         888                     
.d8888b  888888 .d88b.  88888b.  
88K      888   d88""88b 888 "88b 
"Y8888b. 888   888  888 888  888 
     X88 Y88b. Y88..88P 888 d88P 
 88888P'  "Y888 "Y88P"  88888P"  
                        888      
                        888      
                        888      
    
Below here are functions and constants used for the clustering task. 
No need to read or understand them!
"""

GOLD_STANDARD_CLUSTER_STD = 0.2
DIMENSIONS = 32
EPOCHS = 10
N_CLUSTERS = 4
N_SAMPLES = 1000


def generate_data(
    samples: int = N_SAMPLES, dimensions: int = DIMENSIONS, clusters: int = N_CLUSTERS
) -> Tuple[ClusteredDataset, List[Embedding]]:
    """ Generates a random dataset and flat list of embeddings. """
    # Distributions.
    distributions = []
    for _ in range(clusters):
        mean = random.random()
        distributions.append((mean, GOLD_STANDARD_CLUSTER_STD))

    # Generate samples.
    dataset: ClusteredDataset = {}
    embeddings = []
    for _ in range(samples):
        cluster_id = random.randrange(0, clusters)
        mean, std = distributions[cluster_id]
        embedding = tuple(random.normalvariate(mean, std) for _ in range(dimensions))
        dataset.setdefault(cluster_id, []).append(embedding)
        embeddings.append(embedding)
    random.shuffle(embeddings)
    return dataset, embeddings


def evaluate(gold_standard: ClusteredDataset, system: ClusteredDataset):
    """ Calculates and prints evaluation metrics. """

    def get_pairs(dataset: ClusteredDataset) -> Set[Tuple[str, str]]:
        pairs = set()
        for cluster_embeddings in dataset.values():
            pair: Tuple[str, str]
            for pair in itertools.combinations(map(str, cluster_embeddings), 2):
                # Sort for matching stability.
                if pair[0] > pair[1]:
                    pair = (pair[1], pair[0])
                pairs.add(pair)
        return pairs

    gold_pairs = get_pairs(gold_standard)
    system_pairs = get_pairs(system)

    inter = len(gold_pairs.intersection(system_pairs))
    precision = inter / len(system_pairs) if system_pairs else 0
    recall = inter / len(gold_pairs) if gold_pairs else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0
    print(f"Precision {precision:.2f}\tRecall {recall:.2f}\tF1 {f1:.2f}")
