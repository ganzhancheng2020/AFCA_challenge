
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jsonlines
import json
import nest_asyncio
nest_asyncio.apply()


# In[ ]:


# load query
def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


# write result
def write_jsonl(results,submission_id):
    with open(f'result{submission_id}.jsonl', 'w',encoding='utf-8') as outfile:
        for entry in results:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')


# In[ ]:





# In[ ]:




