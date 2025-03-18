import os
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

### merge definitions ###
MERGE_TYPE_LIST = ["static", "adaptive"]
MERGE_METHOD_STATIC_LIST = ["CART", "TSV", "TSVr", "TA", "AVG", "CART_TSV", "TSV_CART", "pre_CART", "TA_CART", "CART_CART"]
MERGE_METHOD_ADAPTIVE_LIST = ["CART", "TA", "TSV"]
MERGE_METHOD_LIST = MERGE_METHOD_STATIC_LIST + MERGE_METHOD_ADAPTIVE_LIST
