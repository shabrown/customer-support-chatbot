import os
import tweepy
from dotenv import load_dotenv
from chatbot import respond

load_dotenv()

CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')
ACCESS_KEY = os.getenv('ACCESS_KEY')
ACCESS_SECRET = os.getenv('ACCESS_SECRET')

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth)

mentions = api.mentions_timeline()

# I use last_seen_id so that Tweepy only return statuses more recent than the specified ID
FILE_NAME = 'last_seen_id.txt'

def retrieve_last_seen_id(file_name):
    f_read = open(file_name, 'r')
    last_seen_id = int(f_read.read().strip())
    f_read.close()
    return last_seen_id


def store_last_seen_id(last_seen_id, file_name):
    f_write = open(file_name, 'w')
    f_write.write(str(last_seen_id))
    f_write.close()
    return


def reply_tweets(event, context):
    # print('retrieving and replying to tweets...', flush=True)
    # DEV NOTE: use 1418636590449037317 for testing.
    last_seen_id = retrieve_last_seen_id(FILE_NAME)
    # NOTE: We need to use tweet_mode='extended' below to show
    # all full tweets (with full_text). Without it, long tweets
    # would be cut off.
    # since_id – Returns only statuses with an ID more recent than the specified ID.
    mentions = api.mentions_timeline(since_id=last_seen_id,
                                     tweet_mode='extended')

    for mention in reversed(mentions):
        last_seen_id = mention.id
        request = mention.full_text[16:]
        store_last_seen_id(last_seen_id, FILE_NAME)

        chatbot_response = respond(f'''{request}''')

        api.update_status(status='@' + mention.user.screen_name + ' ' + chatbot_response,
                          in_reply_to_status_id=mention.id)

