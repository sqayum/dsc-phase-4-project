import tweepy
import os
from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta as delta
from functools import partial


class CustomTweepyClient(tweepy.Client):

    MAX_TWEETS_PER_REQUEST = 100
    MIN_TWEETS_PER_REQUEST = 10

    def __init__(self, bearer_token,  *, tweet_cap, database_path, **kwargs):
        super().__init__(bearer_token, **kwargs)
        self._compute_time_distributed_request_rates(tweet_cap)
        self._compute_time_intervals()

        self._db_path = os.path.join(os.path.dirname(__file__), database_path)
        self._db_conn = None
        self._db_cur = None
        self._initialize_database_connection()

    def _compute_time_distributed_request_rates(self, tweet_cap):
        if (tweet_cap < CustomTweepyClient.MIN_TWEETS_PER_REQUEST):
            raise ValueError("<tweet_cap> PARAMETER TOO SMALL -- AT LEAST 10 TWEETS REQUIRED TO MAKE A REQUEST")
        else:
            self._num_intervals = 7
            tweet_cap_per_interval = tweet_cap // self._num_intervals
            if (tweet_cap_per_interval <= CustomTweepyClient.MAX_TWEETS_PER_REQUEST) and (tweet_cap_per_interval >= CustomTweepyClient.MIN_TWEETS_PER_REQUEST):
                self._num_requests_per_interval = 1
                self._num_tweets_per_request = tweet_cap_per_interval

            elif (tweet_cap_per_interval > CustomTweepyClient.MAX_TWEETS_PER_REQUEST):
                self._num_tweets_per_request = CustomTweepyClient.MAX_TWEETS_PER_REQUEST
                while (tweet_cap_per_interval % self._num_tweets_per_request != 0) and (self._num_tweets_per_request >= CustomTweepyClient.MIN_TWEETS_PER_REQUEST):
                    self._num_tweets_per_request -= 1
                self._num_requests_per_interval = tweet_cap_per_interval // self._num_tweets_per_request

            else:
                while (tweet_cap_per_interval < CustomTweepyClient.MIN_TWEETS_PER_REQUEST):
                    self._num_intervals -= 1
                    tweet_cap_per_interval = tweet_cap // self._num_intervals
                self._num_requests_per_interval = 1
                self._num_tweets_per_request = tweet_cap_per_interval
        print(f"---\nnum_intervals: {self._num_intervals}\nnum_requests_per_interval: {self._num_requests_per_interval}\nnum_tweets_per_request: {self._num_tweets_per_request}\n---")

    def _compute_time_intervals(self):
        self._time_intervals = [[(datetime.now(timezone.utc) + delta(days=-(i+1))).isoformat(timespec="seconds"), (datetime.now(timezone.utc) + delta(days=-i)).isoformat(timespec="seconds")] for i in range(self._num_intervals)][::-1]
        self._time_intervals[0][0] = (datetime.fromisoformat(self._time_intervals[0][0]) + delta(hours=+2)).isoformat(timespec="seconds")
        self._time_intervals[-1][-1] = (datetime.fromisoformat(self._time_intervals[-1][-1]) + delta(hours=-1)).isoformat(timespec="seconds")

    def _initialize_database_connection(self):
        self._db_conn = sqlite3.connect(self._db_path)
        self._db_cur = self._db_conn.cursor()

        self._db_cur.execute("""CREATE TABLE IF NOT EXISTS tweets(
                                    id INTEGER PRIMARY KEY,
                                    user_id INTEGER NOT NULL,
                                    date TEXT NOT NULL,
                                    like_count INTEGER DEFAULT 0,
                                    user_name TEXT,
                                    user_location TEXT,
                                    text TEXT NOT NULL)""")

        self._db_conn.close()
        self._db_conn = None


    @staticmethod
    def _process_response(response):
        tweet_list = response.pop("data")
        reference_tweet_list = response["includes"].pop("tweets")
        for tweet in tweet_list:
            if "referenced_tweets" in tweet:
                for referenced_tweet in tweet["referenced_tweets"]:
                    if referenced_tweet["type"] == "retweeted":
                        reference_id = referenced_tweet["id"]
                        for reference_tweet in reference_tweet_list:
                            if reference_tweet["id"] == reference_id:
                                tweet["text"] = reference_tweet["text"]
                del tweet["referenced_tweets"]

            tweet["user_id"] = tweet.pop("author_id")
            tweet["date"] = tweet.pop("created_at").split('T')[0]
            tweet["like_count"] = tweet["public_metrics"]["like_count"]
            del tweet["public_metrics"]

        user_list = response["includes"].pop("users")
        for user in user_list:
            user["user_name"] = user.pop("name")
            del user["username"]
            if "location" in user:
                user["user_location"] = user.pop("location")
            else:
                user["user_location"] = ""

        return [{**tweet, **user} for tweet, user in zip(tweet_list, user_list)]

    def _search_recent_tweets(self, *args, **kwargs):
        try:
            response = partial(super().search_recent_tweets, media_fields=None, place_fields=None, poll_fields=None, max_results=self._num_tweets_per_request)(*args, **kwargs)
        except tweepy.HTTPException as error:
            if isinstance(error, tweepy.TooManyRequests):
                print("\nERROR: REQUEST RATE EXCEEDED\nPAUSING FOR 15 MINUTES...\n")
                time.sleep(900)
                print("...RESUMING REQUESTS\n")
                return self._search_recent_tweets(*args, **kwargs)
            else:
                print(error)
                return
        return response

    def _append_tweets_to_database(self, response):
        for tweet in response:
            tweet_columns = (tweet["id"],
                                tweet["user_id"],
                                tweet["date"],
                                tweet["like_count"],
                                tweet["user_name"],
                                tweet["user_location"],
                                tweet["text"])

            self._db_cur.execute("""INSERT OR IGNORE INTO tweets(id, user_id, date, like_count, user_name, user_location, text)
                                        VALUES(?,?,?,?,?,?,?)""", tweet_columns)

            self._db_conn.commit()


    def extract_tweets(self, query, *, tweet_fields=None, user_fields=None, expansions=None):
        self._db_conn = sqlite3.connect(self._db_path)
        self._db_cur = self._db_conn.cursor()

        query = ' '.join(query.split())
        for interval in self._time_intervals:
            for _ in range(self._num_requests_per_interval):
                start_time, end_time = interval[0], interval[1]
                response = self._search_recent_tweets(query, expansions=expansions, tweet_fields=tweet_fields, user_fields=user_fields, start_time=start_time, end_time=end_time)
                if response is None:
                    break
                response = self._process_response(response)
                self._append_tweets_to_database(response)

        self._db_conn.close()
        self._db_conn = None




def main():
    BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
    QUERY = '''
              ("joe" OR "president") "biden" OR "biden administration" OR @POTUS OR @JoeBiden
              OR #biden OR #joebiden OR #POTUS
              -is:quote
              lang:en
              '''

    TWEET_FIELDS = ["id", "author_id", "created_at", "public_metrics", "referenced_tweets", "text"]
    USER_FIELDS = ["location"]
    EXPANSIONS = ["author_id", "referenced_tweets.id"]
    DATABASE_PATH = "data/tweets.db"


    client = CustomTweepyClient(BEARER_TOKEN,
                                        tweet_cap=500_000,
                                        database_path=DATABASE_PATH,
                                        return_type=dict)


    client.extract_tweets(query=QUERY,
                                tweet_fields=TWEET_FIELDS,
                                user_fields=USER_FIELDS,
                                expansions=EXPANSIONS)


if __name__ == "__main__":
    main()