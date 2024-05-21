import json
import re
from collections import defaultdict
from tqdm import tqdm

# List of assets
asset_list = [
    "amazon",
    "primeday", 
    "alexa", 
    "kindle", 
    "aws",
    "amzn",
    "prime",
    "bezos",
    "e-commerce giant"
]

# Precompile regex patterns for each asset
asset_patterns = {asset: re.compile(r'\b{}\b'.format(re.escape(asset)), re.IGNORECASE) for asset in asset_list}

def extract_asset_mentions(comment):
    """
    Extract mentions of assets in the comment.
    """
    asset_mentions = defaultdict(list)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', comment)
    for sentence in sentences:
        for asset, pattern in asset_patterns.items():
            if pattern.search(sentence):
                asset_mentions[asset].append(sentence)
    return asset_mentions

def process_comment(comment):
    """
    Process a single comment to extract asset mentions.
    """
    comment_dict = extract_asset_mentions(comment)

    if len(comment_dict) == 0:
        return None
                
    # if len(comment_dict) == 1:
    #     # if only one asset is mentioned, return the whole comment
    #     return {list(comment_dict.keys())[0]: comment}
    
    # if multiple assets are mentioned, return the sentence that mentions each asset
    filtered_dict = {}
    for asset, sentences in comment_dict.items():
        filtered_dict[asset] = " ".join(sentences)

    return filtered_dict

def process_batch(batch, asset_name):
    """
    Process a batch of comments.
    """
    asset_comments = defaultdict(list)
    for d in batch:
        comment = f"{d['title']} {d['text']}"

        # Some preprocessing
        comment = re.sub(r'http\S+', '', comment)
        comment = comment.replace('\n', ' ')
        comment = re.sub(' +', ' ', comment)

        filtered_dict = process_comment(comment)

        # concatenate comments for each asset
        if filtered_dict:
            text = " ".join([v for v in filtered_dict.values()])
            asset_comments[asset_name].append({
                'date': d['date'],
                'text': text
            })

    return asset_comments

def filter_dataset(data_path, asset_name):
    """
    Filter the dataset to only keep the comments that mention one of the assets in asset_labels.
    """
    asset_comments = defaultdict(list)
    chunk_size = 100

    with open(data_path, 'r') as file:
        data = json.load(file)
        total_lines = len(data)
        print("Finished loading data")
        for i in tqdm(range(0, total_lines, chunk_size)):
            batch = data[i:i + chunk_size]
            comments = process_batch(batch, asset_name)
            for _, comment in comments.items():
                asset_comments[asset_name].extend(comment)

    return asset_comments

def save_asset_comments(asset_comments, output_path):
    """
    Save the extracted asset comments to a JSON file.
    """
    with open(output_path, 'w') as file:
        json.dump(asset_comments, file, indent=4)

if __name__ == "__main__":
    data_paths = ["daytr.json", "inv.json", "stocks.json", "wsb_cleaned.json"]
    asset_comments = defaultdict(list)

    for data_path in data_paths:
        temp_dict = filter_dataset(f"Data/comments/{data_path}", asset_name="amazon")
        for asset, comments in temp_dict.items():
            asset_comments[asset].extend(comments)

    for asset, comments in asset_comments.items():
        # Write comments to file
        asset_name = asset.replace(" ", "_")
        asset_name = asset_name.replace("/", "_")
        asset_name = asset_name.replace("(", "")
        asset_name = asset_name.replace(")", "")
        asset_name = asset_name.replace("&", "")
        output_path = f"Data/processed/{asset_name}_comments.json"
        save_asset_comments(comments, output_path)
