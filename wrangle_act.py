#!/usr/bin/env python
# coding: utf-8

# # Gather

# In[75]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import json
import os
import io
import re


# In[76]:


# Open Archive data
df_archive = pd.read_csv('twitter-archive-enhanced.csv')


# In[77]:


# Open tweets JSON data in dataframe:
info_list = []
with open('tweet_json.txt', encoding= 'utf-8') as fh:
    for line in fh:
        tweet = json.loads(line)
        tweet_id = tweet["id"]
        favorite_count = tweet["favorite_count"]
        retweet_count = tweet["retweet_count"]
        
        info_list.append({"tweet_id": tweet_id, "retweet_count": retweet_count, "favorite_count": favorite_count})

df_tweets = pd.DataFrame(info_list)


# In[78]:


# Open image_predictions in dataframe:
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
filename = url.split('/')[-1]
response = requests.get(url)
if not os.path.isfile(filename):
    with open(filename , 'wb') as fh:
        fh.write(response.content)
#df_predict = pd.read_csv(io.StringIO(response), sep = '\t')
df_predict = pd.read_csv(filename, sep = '\t')


# In[79]:


# Opening deleted_ids list file gatherd from tweets_json API code
with open('deleted_ids.txt', encoding= 'utf-8') as file:
    for line in file:
        deleted_ids = json.loads(line)
type(deleted_ids)


# # Assess

# In[80]:


df_archive.head()


# In[81]:


df_archive.info()


# In[82]:


print(df_archive.name.value_counts())
df_archive.name.unique()


# In[83]:


df_archive['rating_denominator'].value_counts()


# In[84]:


df_archive['rating_numerator'].value_counts()


# In[85]:


df_archive['puppo'].value_counts()


# In[86]:


df_tweets.info()


# In[87]:


df_tweets.head()


# In[88]:


df_predict.info()


# In[89]:


df_predict.sample(10)


# ## ASSESS Summary
# #### Quality
#         df_archive:
#             * Unoriginal ratings (retweeted) 
#             * Some tweets don't have images(replies and missing expanded urls)
#             * Some rows of the tweet_id column are deleted ids
#             * Some dogs are named "None"
#             * Replace dogs Stages 'None' To NaN
#             * Rating numerator type should be Float
#             * 'tweet_id' in 3 tables are type integer
#             * 'timestamp' column is an oject type and contains both time and date
#             * Some names are extracted wrong [a, an, one, not]
#             * Some rating_denominator are extracted wrong [0,2,7]
#             * Some rating_numerator are extracted wrong
# #### Tidiness
#             * Dogs stages columns should be one column.
#             * Source column is not needed
#             * add favourite and retweets counts to the archive dataframe
#             

# # Clean

# In[90]:


archive_clean = df_archive.copy()
predict_clean = df_predict.copy()
tweets_clean = df_tweets.copy()


# ## Quality
# #### Define: 
#             Remove invalid rows of data 
#                     Archive: drop retweeted, replies, deleted tweets and rows with no pictures(missing expanded urls).
# #### Code

# In[91]:


# Remove retweeted rows
retweeted = archive_clean[~archive_clean['retweeted_status_id'].isnull()].index
archive_clean = archive_clean.drop(retweeted, axis=0)
# Drop retweeted info columns
archive_clean = archive_clean.drop(['retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp'], axis =1)


# In[92]:


# Remove replies rows
replies = archive_clean[~archive_clean['in_reply_to_status_id'].isnull()].index
archive_clean.drop(replies, axis=0, inplace = True)
# Drop replies info columns
archive_clean.drop(['in_reply_to_status_id', 'in_reply_to_user_id'], axis = 1, inplace = True)


# In[93]:


# Remove deleted status rows
for entry in deleted_ids:
    archive_clean.drop(archive_clean[archive_clean['tweet_id']==entry].index, axis = 0 , inplace= True)


# In[94]:


# Remove missing expanded_url rows andthen dropping the column
archive_clean.drop(archive_clean[archive_clean.expanded_urls.isnull()].index, axis=0, inplace= True)
archive_clean.drop('expanded_urls', axis = 1 , inplace = True)


# In[95]:


# resetting index
archive_clean.reset_index(inplace = True)


# In[96]:


archive_clean.drop(['index'], axis=1, inplace = True)


# #### Test

# In[97]:


archive_clean.info()


# In[98]:


archive_clean.shape


# #### Define
#             Replace 'None' values in name and category columns to NaN using numpy library and replace() function
# #### Code

# In[99]:


archive_clean.name.replace({'None': np.nan}, inplace = True)


# #### Test

# In[100]:


archive_clean.info()


# #### Test

# #### Define 
#             Change rating numerator type using astype()
#             Change 'tweet_id' type to 'string' in the 3 dataframes
# #### Code

# In[101]:


archive_clean['rating_numerator'] = archive_clean['rating_numerator'].astype('float')
#archive_clean['rating_denominator'] = archive_clean['rating_denominator'].astype('int')
archive_clean.tweet_id = archive_clean.tweet_id.astype('string')
predict_clean.tweet_id = predict_clean.tweet_id.astype('string')
tweets_clean.tweet_id = tweets_clean.tweet_id.astype('string')


# #### Test

# In[102]:


archive_clean.info()


# #### Define 
#             Extract date part from 'timestamp' to a new column 'date' using string slicing and change type to datetime
#             and the drop 'timestamp' column.
#             Extract 'year', 'month', 'day' to new columns
#             Rearrange columns to put date column after 'tweet_id'
#             
# #### Code

# In[103]:


archive_clean['date'] = archive_clean['timestamp'].str[0:10]
archive_clean['date'] = pd.to_datetime(archive_clean['date'])
archive_clean.drop('timestamp', axis = 1, inplace= True)


# In[104]:


archive_clean['year'] = archive_clean.date.dt.year.astype('string')
archive_clean['month'] = archive_clean.date.dt.month.astype('string')
archive_clean['day'] = archive_clean.date.dt.day.astype('string')


# In[105]:


archive_clean = archive_clean.reindex(columns=['tweet_id', 'date', 'year', 'month', 'day', 'source', 'text', 'rating_numerator', 'rating_denominator', 'name', 'doggo', 'floofer', 'pupper', 'puppo'])


# #### Test

# In[106]:


archive_clean.info()


# #### Define
#         Fix wrong extracted names [a, an, not, one] by extracting the correct name using RE and if not found put NaN
# #### Code

# In[107]:


pattern = re.compile(r'(?:name(?:d)?)\s{1}(?:is\s)?([A-Za-z]+)')
for index, row in archive_clean.iterrows():  
    try:
        if row['name'] == "a":
            c_name = re.findall(pattern, row['text'])[0]
            archive_clean.loc[index,'name'] = archive_clean.loc[index,'name'].replace('a', c_name)
        elif row['name'] == 'an':
            c_name = re.findall(pattern, row['text'])[0]
            archive_clean.loc[index,'name'] = archive_clean.loc[index,'name'].replace('an', c_name)
        elif row['name'] == 'not':
            c_name = re.findall(pattern, row['text'])[0]
            archive_clean.loc[index,'name'] = archive_clean.loc[index,'name'].replace('an', c_name)
        elif row['name'] == 'one':
            c_name = re.findall(pattern, row['text'])[0]
            archive_clean.loc[index,'name'] = archive_clean.loc[index,'name'].replace('an', c_name)
    except IndexError:
        archive_clean.loc[index,'name'] = np.nan

            
archive_clean.name.value_counts(dropna = False)


# In[108]:


'a' in archive_clean.name.unique()


# #### Define
#             Fix wrong extracted denominators
#             Fix wrong extracted numerators
# #### Code

# In[109]:


archive_clean[archive_clean.text.str.contains(r"(\d+\.\d*\/\d+)")][['text', 'rating_denominator']]

archive_clean[archive_clean.text.str.contains(r"(\d+\.\d*\/\d+)")][['text', 'rating_denominator']].replace(archive_clean.rating_denominator, archive_clean.text.str.contains(r"(\d+\.\d*\/\d+)"), regex=True, inplace=True)


# In[110]:


archive_clean[archive_clean.text.str.contains(r"(\d+\.\d*\/\d+)")][['text', 'rating_numerator']]

archive_clean[archive_clean.text.str.contains(r"(\d+\.\d*\/\d+)")][['text', 'rating_numerator']].replace(archive_clean.rating_numerator, archive_clean.text.str.contains(r"(\d+\.\d*\/\d+)"), regex=True, inplace=True)


# #### Test

# In[111]:


archive_clean.rating_denominator.value_counts()


# In[112]:


archive_clean.rating_numerator.value_counts()


# #### Define
#             Remove rows that are deleted statuses from 'predict_clean'
# #### Code

# In[113]:


for entry in deleted_ids:
    predict_clean.drop(predict_clean[predict_clean['tweet_id']==entry].index, axis = 0 , inplace= True)


# ##### Test

# In[114]:


predict_clean.info()


# ## Tidiness
# 
# #### Define
#             Replace 'None' in stages columns with '' and then concatenate them in a single column named "dog_stage", then strip it and replace '' with np.nan and then edit manually the mixed stages and drop the four columns.
# #### Code

# In[115]:


archive_clean['doggo'].replace({'None': ''}, inplace= True)
archive_clean['floofer'].replace({'None': ''}, inplace= True)
archive_clean['pupper'].replace({'None': ''}, inplace= True)
archive_clean['puppo'].replace({'None': ''}, inplace= True)


# In[116]:


archive_clean['dog_stage']= archive_clean.doggo + archive_clean.floofer + archive_clean.pupper + archive_clean.puppo


# In[117]:


archive_clean.dog_stage = archive_clean.dog_stage.str.strip()


# In[118]:


archive_clean.replace({'': np.nan}, inplace= True)


# In[119]:


archive_clean.dog_stage.unique()


# In[120]:


archive_clean.dog_stage.replace({'doggopuppo': 'doggo-puppo'}, inplace = True)
archive_clean.dog_stage.replace({'doggofloofer': 'doggo-floofer'}, inplace = True)
archive_clean.dog_stage.replace({'doggopupper': 'doggo-pupper'}, inplace = True)
archive_clean.drop(['doggo', 'floofer', 'pupper', 'puppo'], axis =1 , inplace= True)


# #### Test

# In[121]:


archive_clean.dog_stage.value_counts(dropna= False)


# In[122]:


archive_clean.info()


# #### Define
#         Dropping not needed 'source' column from 'archive_clean'
# #### Code

# In[123]:


archive_clean.drop('source', axis = 1, inplace = True)


# #### Test

# In[124]:


archive_clean.info()


# #### Define
#             Merging the two dfs 'archive_clean' and 'df_tweets' on tweet_id from the first.
# #### Code

# In[125]:


archive_clean1 = archive_clean.merge(tweets_clean, on = 'tweet_id', how = 'inner')


# #### Test

# In[126]:


archive_clean1.info()


# In[127]:


archive_clean1.sample(10)


# # Store

# In[128]:


archive_clean1.to_csv('twitter_archive_master.csv', index = False)


# # Insights

# In[129]:


# 1
archive_clean1.groupby(by ='dog_stage', dropna= False).mean()


# In[130]:


# 2
archive_clean1.groupby(['year', 'month'], sort = False)[['retweet_count', 'favorite_count']].mean()


# In[131]:


# 3
archive_clean1.query('rating_numerator < 10').favorite_count.mean()


# In[132]:


archive_clean1.query('rating_numerator >= 10').favorite_count.mean()


# In[133]:


# 4
archive_clean1.boxplot([ 'rating_numerator', 'rating_denominator'])


# In[134]:


archive_clean1[archive_clean1['rating_numerator']<300].boxplot([ 'rating_numerator', 'rating_denominator'])


# # Visualization

# In[135]:


plt.bar(archive_clean1.year, archive_clean1.favorite_count)
plt.title('Total Favorite counts across years', fontsize= 15)
plt.xlabel('Year', fontsize= 12)
plt.ylabel('Favorite counts', fontsize= 12)


# In[136]:


tweets_per_day = archive_clean1.groupby('date', sort = False)[['tweet_id']].count()
likes_per_tweet_per_day = archive_clean1.groupby('date', sort = False)[['favorite_count']].mean()


# In[137]:


plt.scatter(tweets_per_day, likes_per_tweet_per_day)
plt.title('Fav counts and tweeting frequency corelation', fontsize= 15)
plt.xlabel('Tweets per day', fontsize = 12)
plt.ylabel('Mean fav counts per tweet', fontsize = 12)


# In[ ]:




