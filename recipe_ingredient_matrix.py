import click
import os
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import concatenate_tags_of_item

# Home_DIR
HOME_DIR = os.path.expanduser("~/.datasets/data/")

# Read dataframes
df_ratings = pd.read_csv(HOME_DIR + "ratings.csv")
df_ingredients_ids = pd.read_csv(HOME_DIR + "ingredients_ids.csv")
df_ingredients = pd.read_csv(HOME_DIR + "ingredients.csv")
df_recipes = pd.read_csv(HOME_DIR + "recipes_ids.csv")

# Apply the necessary data merges
df_recipe_ingredient = pd.merge(df_ingredients, df_ingredients_ids,
                                on="ingredient_name", how="left")  # We need a dataset with recipe_id and ingredient_id

# Encode features
df_recipe_ingredient['ingredient_id'] = df_recipe_ingredient.ingredient_id.astype(str)
# GroupBy tags by item
df_ingredient_per_recipe = df_recipe_ingredient.groupby('recipe_id')['ingredient_id'].agg(concatenate_tags_of_item)
df_ingredient_per_recipe.name = "recipe_ingredients"
df_ingredient_per_recipe = df_ingredient_per_recipe.reset_index()
df_data = pd.merge(df_recipes, df_ingredient_per_recipe, how="right", on="recipe_id")  # merge to add recipe_name


@click.command()
@click.option('--sample/--no-sample', type=bool, default=False, help="allow sample from the initial dataset")
@click.option('--sample-size', default=15000, type=int, help="size to sample from the dataset")
def generate(sample, sample_size):
    if sample:
        df_data_used = df_data.iloc[list(range(sample_size))]
    else:
        df_data_used = df_data
    tf_idf = TfidfVectorizer()
    df_recipes_tf_idf_described = tf_idf.fit_transform(df_data_used.recipe_ingredients)
    save_npz(HOME_DIR + "recipe_ingredient_matrix.npz", df_recipes_tf_idf_described)
    df_data_used.to_csv(HOME_DIR + "df_data_used.csv", index=False)


if __name__ == '__main__':
    generate()
