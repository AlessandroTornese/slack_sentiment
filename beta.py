#!/usr/bin/env python
# coding: utf-8

# costants

# In[2]: keys are stored in a .env file 


SLACK_BOT_TOKEN ="TOKEN"


host ="SERVER"
database="DBNAME"
user="USER_ID"
password= "PASSWORD"


apikey='APIKEY'
ta_url= 'ENDPOINT'

channel_id ='CHANNEL'


# Importing modules
# 

# In[3]:


import logging
import os
# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)
import numpy as np
import slack_sdk

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pandas as pd
import json

from dotenv import load_dotenv
load_dotenv()

import psycopg2
import psycopg2.extras as extras

#importing ibm watson ToneAnalyzerV3 and authenticator
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators  import IAMAuthenticator
#setting key and url (to be moved to the .env file with all the other credentials)
authenticator= IAMAuthenticator(apikey)
ta= ToneAnalyzerV3(version ='2017-09-21', authenticator = authenticator)
ta.set_service_url(ta_url)


# Connecting to the slack channel 

# In[4]:


# WebClient insantiates a client that can call API methods
# When using Bolt, you can use either `app.client` or the `client` passed to listeners.
client = WebClient(token= SLACK_BOT_TOKEN)

logger = logging.getLogger(__name__)
# Store conversation history
conversation_history = []
# ID of the channel you want to send the message to
channel_id = "C029R2BR8GJ"   #os.getenv("channel_id")

try:
    # Call the conversations.history method using the WebClient
    # conversations.history returns the first 100 messages by default
    # These results are paginated, see: https://api.slack.com/docs/pagination
    result = client.conversations_history(channel=channel_id)


    conversation_history = result["messages"]



    # Print results
    logger.info("{} messages found in {}".format(len(conversation_history), id))

except SlackApiError as e:
    logger.error("Error creating conversation: {}".format(e))


# In[5]:


conversation_history


# In[66]:


df = pd.DataFrame(conversation_history)


# In[7]:


timestamps =list(df['ts'])


# In[8]:


len(timestamps)


# looping through the df to get the text 

# In[9]:


y = range(len(timestamps))
z = 0
pdList =[]
for x in y:
    reply = client.conversations_replies(channel = channel_id, ts = timestamps[z])
    replies= reply['messages']
    chat = pd.json_normalize(replies)
    pdList.append(chat)
    chat_concat = pd.concat(pdList, ignore_index=True)
    z+=1


# In[10]:


#chat_concat is the DataFrame containing all the unnested content from a give channel_id
#removing unnecessary strings, 'has joined'
chat_concat = chat_concat[chat_concat['text'].str.contains("<|> has joined the channel")== False]
#removing text that is not user generated
chat_concat= chat_concat[chat_concat.user.notnull()]
chat_concat


# In[67]:


ts= chat_concat[['ts', 'text','user']] #creating this dataframe for later use to bring timestamps to the sentiment df
# filtering from df only text and users for feeding Tone Analyzer 
chats_text_users = chat_concat[['text', 'user']]  

chats = chats_text_users[chats_text_users.user.notnull()]
chats


# In[12]:


chats = pd.DataFrame(chats)
users = set(chats.user) #getting distinct users for later join 
chats = chats.to_dict('records')


# In[37]:


res = ta.tone_chat(chats).get_result()
res = res["utterances_tone"]


# In[125]:


#flattening the nesting dictionary to make it tabular 
sentiment =pd.json_normalize(res, meta=['utterance_id','utterance_text'], record_path=['tones'])
sentiment.head(15)


# In[101]:


merged = pd.merge(sentiment, ts, how = 'left', left_on='utterance_text', right_on='text')


# In[123]:


merged['datetime'] = pd.to_datetime(merged['ts'], unit= 's')
merged['date'] = pd.to_datetime(merged['datetime'].dt.date)


# In[124]:


merged


# ## Generating User df

# In[16]:


users = list(users)
user_details = []

for user_id in users:
    try:
        # Call the users.info method using the WebClient
        result = client.users_info(
            user= user_id
        )
        logger.info(result)
        
    except SlackApiError as e:
        logger.error("Error fetching conversations: {}".format(e))
    for res in result:
        user_details.append(result['user'])
user_details
    


# In[126]:


users_df =pd.json_normalize(user_details)
users_df 


# In[ ]:

param_dic = {
    "host"      : os.getenv("host"),
    "database"  : os.getenv("database"),
    "user"      : os.getenv("user"),
    "password"  : os.getenv("password")
}


def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1)
    print("Connection successful")
    return conn
conn = connect(param_dic)


# DEFINING A FUNCTION TO RUN SQL SCRIPTS

# In[13]:


def execute_query(conn, query):
    """ Execute a single query """

    ret = 0 # Return value
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1

    # If this was a select query, return the result
    if 'select' in query.lower():
        ret = cursor.fetchall()
    cursor.close()
    return ret


# ## DROP TABLE IN POSTGRE BTEX DATABASE if already exists

# In[18]:


execute_query(conn, "drop table if exists public.slack_sentiment;")
print('Table dropped')


# In[19]:


# CREATE a NEW TABLES if not exists (currently commented out)
 
execute_query(conn, ''' 
CREATE TABLE IF NOT EXISTS public.slack_sentiment (
   SCORE float not NULL,
   tone_id VARCHAR (255) NULL,
   tone_name VARCHAR (255) NULL,
   utterance_id int not NULL,
   utterance_text VARCHAR (15000) NULL
);''') 
 
print('the slack_sentiment table has been created')


# ## definining a function that fetches the dataframe and inserts it into the postgre table

# In[20]:


def execute_values(conn, df, table):
    """
    Using psycopg2.extras.execute_values() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query  = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("Data upload done")
    cursor.close()




# In[21]:


#get_ipython().run_line_magic('time', '')
# push the data into the db
execute_values(conn, df, 'slack_sentiment')

