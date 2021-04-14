import os
import click
import pandas as pd

# Home_DIR
HOME_DIR = os.path.expanduser("~/.datasets/data/")

# Load similarity recipes matrix
similarity_matrix = pd.read_pickle(HOME_DIR + "similarity_recipes_matrix.pkl")


@click.command()
@click.argument('recipe-id')
@click.argument('recommended-size')
def recommend(recipe_id, recommended_size):
    recommended_column = similarity_matrix.loc[int(recipe_id)].sort_values(ascending=False)[:int(recommended_size)]
    recommended_data = pd.DataFrame(recommended_column)
    recommended_data.to_csv(HOME_DIR + str(recipe_id) + "_" + "first_" + str(recommended_size) + "_recommendations.csv")


if __name__ == '__main__':
    recommend()
