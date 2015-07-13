import datetime
import json
import sys

import pandas as pd
import tweepy

from candidates import candidates
from private_settings import consumer_key, consumer_secret, access_token, access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)
cursor = lambda term: tweepy.Cursor(api.search, q = term, rpp = 100)


def tweet_to_dict(tweet, candidate):
    return {
        'candidate': candidate,
        'id': tweet.id,
        'coordinates': tweet.coordinates,
        'favorite_count': tweet.favorite_count,
        'created_at': tweet.created_at,
        'geo': tweet.geo,
        'lang': tweet.lang,
        'place': tweet.place,
        'retweet_count': tweet.retweet_count,
        'text': tweet.text,
        'user_location': tweet.user.location,
        'user_name': tweet.user.name,
        'user_screen_name': tweet.user.screen_name,
        'user_time_zone': tweet.user.time_zone,
        'user_followers_count': tweet.user.followers_count
    }


def tweets_json(tweets):
    return [tweet._json for tweet in tweets]


def tweets_df(tweets, term, f):
    return pd.DataFrame([f(tweet, term) for tweet in tweets])


def search(cursor, term, number):
    return list(cursor(term).items(number))


if __name__ == '__main__':
    number_per_candidate = int(sys.argv[1])

    dfs = []
    jsons = {}

    for candidate in candidates:
        print('Now searching for ' + candidate)
        tweets = search(cursor, candidate, number_per_candidate)
        dfs.append(tweets_df(tweets, candidate, tweet_to_dict))
        jsons[candidate] = tweets_json(tweets)

    df = pd.concat(dfs)
    postfix = str(datetime.datetime.now())

    df.to_csv('data/results-{}.csv'.format(postfix))
    with open('data/results-{}.json'.format(postfix), 'w') as json_file:
        json_file.write(json.dumps(jsons))
