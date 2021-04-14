import os
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Home_DIR
HOME_DIR = os.path.expanduser("~/.datasets/data/")

# Load recipe_ingredient matrix and df_data_used
recipe_ingredient_matrix = load_npz(HOME_DIR + "recipe_ingredient_matrix.npz")
df_data_used = pd.read_csv(HOME_DIR + "df_data_used.csv")

# Compute similarity and build matrix
m2m = cosine_similarity(recipe_ingredient_matrix)
df_tfidf_m2m = pd.DataFrame(m2m)

# Index and columns
index_to_recipe_id = df_data_used['recipe_id']
df_tfidf_m2m.columns = [str(index_to_recipe_id[int(col)]) for col in df_tfidf_m2m.columns]
df_tfidf_m2m.index = [index_to_recipe_id[idx] for idx in df_tfidf_m2m.index]

# Save matrix
df_tfidf_m2m.to_pickle(HOME_DIR + "similarity_recipes_matrix.pkl")
