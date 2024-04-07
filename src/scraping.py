import praw
import datetime as dt
import json
import time
from dotenv import dotenv_values
from tqdm import tqdm

# Load environment variables from .env file
config = dotenv_values("variables.env")

class RedditScrapper:
    def __init__(self, client_id, client_secret, user_agent, username):
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent,
                                  username=username)

    def scrape_subreddit(self, subreddit_name, limit=None):
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        save_interval = 100  # Number of posts to scrape before saving to file

        # Initialize tqdm progress bar
        for submission in tqdm(subreddit.new(limit=limit), total=limit, desc="Scraping posts"):
            post_data = {
                'title': submission.title,
                'text': submission.selftext,
                'date': dt.datetime.fromtimestamp(submission.created_utc).isoformat()
            }
            posts.append(post_data)

            if len(posts) >= save_interval:
                self.append_to_json(posts, 'Data/wsb_posts.json')
                posts.clear()  # Clear the list after saving

            # Sleep to ensure compliance with Reddit's rate limit
            time.sleep(1)

        # Append any remaining posts
        if posts:
            self.append_to_json(posts, 'wsb_posts.json')

    def append_to_json(self, data, file_name):
        try:
            with open(file_name, 'r+', encoding='utf-8') as file:
                # First we load existing data into a dict.
                file_data = json.load(file)
                # Join new_data with file_data
                file_data.extend(data)
                # Sets file's current position at offset.
                file.seek(0)
                # Convert back to json.
                json.dump(file_data, file, ensure_ascii=False, indent=4)
        except FileNotFoundError:
            # If the file does not exist, create it and write the data
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # load environment variables
    client_id = config['REDDIT_CLIENT_ID']
    client_secret = config['REDDIT_CLIENT_SECRET']
    user_agent = config['REDDIT_USER_AGENT']
    username = config['REDDIT_USERNAME']

    # You need to replace 'your_client_id', 'your_client_secret', and 'your_user_agent' with your actual Reddit API credentials
    scrapper = RedditScrapper(client_id, client_secret, user_agent, username)

    # Since we can't directly query posts by date, you might limit the number of posts or manually filter by date after fetching
    posts = scrapper.scrape_subreddit('wallstreetbets', limit=100000)
    scrapper.save_to_json(posts, 'wsb_posts.json')
