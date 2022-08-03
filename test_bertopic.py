
#%%
# by default install torch (cpu)
# go to 
# poetry shell
# pip uninstall torch torchvision
# pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

from dataclasses import dataclass
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

from datetime import datetime


 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

#%%
start = datetime.now()
topic_model = BERTopic()
t1 = datetime.now()
print(f"Bertopic init={t1-start}")

topics, probs = topic_model.fit_transform(docs)
t2= datetime.now()
print(f"Bertopic fit={t2-t1}")

#%%
topic_model.get_topic_info()

#%%

topic_model.get_topic(0)

#%%
topic_model.visualize_topics()

#%%

