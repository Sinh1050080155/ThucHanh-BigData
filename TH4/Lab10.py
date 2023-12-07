from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
 
 
twitter = """
Twitter is an online social media and social networking service owned and operated by American company X Corp.,
the legal successor of Twitter, Inc. Twitter users outside the United States are legally served by the Ireland-based
Twitter International Unlimited Company, which makes these users subject to Irish and European Union data protection laws.
On Twitter users post texts, photos and videos known as 'tweets'. Registered users can tweet, like, 'retweet' tweets,
and direct message (DM) other registered users, while unregistered users only have the ability to view public tweets.
Users interact with Twitter through browser or mobile frontend software, or programmatically via its APIs.
"""
facebook = """
Facebook is an online social media and social networking service owned by American technology giant Meta Platforms.
Created in 2004 by Mark Zuckerberg with fellow Harvard College students and roommates Eduardo Saverin, Andrew McCollum,
Dustin Moskovitz, and Chris Hughes, its name derives from the face book directories often given to American university students.
Membership was initially limited to only Harvard students, gradually expanding to other North American universities and,
since 2006, anyone over 13 years old. As of December 2022, Facebook claimed 2.96 billion monthly active users, and ranked third
worldwide among the most visited websites. It was the most downloaded mobile app of the 2010s. Facebook can be accessed from devices
with Internet connectivity, such as personal computers, tablets and smartphones. After registering, users can create a profile
revealing information about themselves. They can post text, photos and multimedia which are shared with any other users who have
agreed to be their friend' or, with different privacy settings, publicly. Users can also communicate directly with each other with
Messenger, join common-interest groups, and receive notifications on the activities of their Facebook friends and the pages they follow.
"""
tiktok = """
TikTok, and its Chinese counterpart Douyin (Chinese: 抖音; pinyin: Dǒuyīn), is a short-form video hosting service owned by ByteDance.
It hosts user-submitted videos, which can range in duration from 3 seconds to 10 minutes. Since their launches, TikTok and Douyin have
gained global popularity.[6][7] In October 2020, TikTok surpassed 2 billion mobile downloads worldwide. Morning Consult named TikTok the
third-fastest growing brand of 2020, after Zoom and Peacock. Cloudflare ranked TikTok the most popular website of 2021,
surpassing google.com.
"""
instagram = """
Instagram is a photo and video sharing social networking service owned by American company Meta Platforms. The app allows users to
upload media that can be edited with filters and organized by hashtags and geographical tagging. Posts can be shared publicly or
with preapproved followers. Users can browse other users' content by tag and location, view trending content, like photos, and follow
other users to add their content to a personal feed. Instagram was originally distinguished by allowing content to be framed only in a
square (1:1) aspect ratio of 640 pixels to match the display width of the iPhone at the time. In 2015, this restriction was eased with
an increase to 1080 pixels. It also added messaging features, the ability to include multiple images or videos in a single post, and a
Stories feature—similar to its main competitor Snapchat—which allowed users to post their content to a sequential feed, with each post
accessible to others for 24 hours. As of January 2019, Stories is used by 500 million people daily.
"""
 
 
 
 
documents = [twitter, facebook, tiktok, instagram]
count_vectorizer = CountVectorizer(stop_words="english")
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)
 
 
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(
   doc_term_matrix,
   columns=count_vectorizer.get_feature_names_out(),
   index=["twitter", "facebook", "tiktok", "instagram"],
)
print(df)
print(cosine_similarity(df, df))
