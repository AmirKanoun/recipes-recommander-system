

# Function to concatenate tags by item
def concatenate_tags_of_item(tags):
    tags_as_str = ' '.join(set(tags))
    return tags_as_str
