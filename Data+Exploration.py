
# coding: utf-8

# In[1]:


import pandas as pd
import os 


# In[ ]:



        
        
        
        


# In[25]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

tickers = ['MSFT', 'GOOG', 'IBM']
days = ['today', 'tomorrow', 'day_after_tomorrow']
width = 0.3
for ix, f in enumerate(tickers):
    df = pd.read_csv(os.path.join("./data/all_data", "combined_{}_tech_news.csv".format(f)), sep='\t')
    bar = df['today'].value_counts().as_matrix()
    bar = bar/np.sum(bar.astype(np.float32))
    print bar
    mu = df.describe()['today']['mean']
    variance = df.describe()['today']['std']
    sigma = math.sqrt(variance)
    x = np.linspace(mu-3*variance,mu+3*variance, 100)
    #plt.plot(x,mlab.normpdf(x, mu, sigma), label=f)
    plt.bar([-0.3 + (ix * width), 0.7 + (ix * width)], bar,width=width, label=f)
    plt.xticks((0,1))
    plt.title('Stock Distribution')
plt.legend(loc='upper right')
plt.show()


# In[ ]:




