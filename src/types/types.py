from typing import List, Dict, Any, Tuple

# Define a type for product information
ProductInfo = Dict[str, Any]

# Define a type for embeddings
Embeddings = List[float]

# Define a type for recommendation results
RecommendationResult = Tuple[List[str], List[float]]  # List of ASINs and their corresponding scores

# Define a type for the configuration settings
Config = Dict[str, Any]