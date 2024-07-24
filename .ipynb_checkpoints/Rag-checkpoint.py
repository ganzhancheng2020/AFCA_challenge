
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ReadLoad import read_jsonl, write_jsonl, read_json, write_json
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document
from tqdm import tqdm

test_dataB = read_jsonl("dataset/test-B-0722.json")


# In[2]:


documents = [Document(text=t['条款'],
                      metadata={
                          '产品名': t['产品名'],
                          'ID': t['ID'],
                          '问题': t['问题']
                      }) for t in test_dataB]
for data in tqdm(documents):
    data.text = data.text.replace('。','. ')
    data.text = data.text.replace('！','! ')
    data.text = data.text.replace('？','? ')
    
# documents[:5]


# In[3]:


from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document

parser = SentenceWindowNodeParser.from_defaults(
    # how many sentences on either side to capture
    window_size=3,
    # the metadata key that holds the window of surrounding sentences
    window_metadata_key="window",
    # the metadata key that holds the original sentence
    original_text_metadata_key="original_sentence"
)
nodes = parser.get_nodes_from_documents(documents[:5])


# In[4]:


from modelscope import snapshot_download
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_dir = snapshot_download('Xorbits/bge-small-zh-v1.5')
embeding = HuggingFaceEmbedding(
        model_name=model_dir,
        cache_folder="./",
        embed_batch_size=512,
    )
Settings.embed_model = embeding


# In[5]:


from llama_index.core import VectorStoreIndex
index = VectorStoreIndex(nodes)


# In[6]:


from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
import jieba


def chinese_tokenizer(text: str) -> list[str]:
    tokens = jieba.lcut(text)
    # TOOD: 短语不可分割
    # TODO: remove stopwords
    return tokens


def retrieve_clause(query):
    #vector retriever
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="问题", value=query
            ),
        ]
    )
    vector_retriever = index.as_retriever(similarity_top_k=2, filters=filters)
    retrieve_nodes = vector_retriever.retrieve(query)
    
    #BM25 retriever
    filter_nodes = [node for node in nodes if node.metadata['问题']==query]
    if len(filter_nodes) == 0:
        filter_nodes = nodes

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=filter_nodes,  # 使用从 VectorStoreIndex 获取的文档存储
        similarity_top_k=2,        # 返回最相似的前3个文档
        tokenizer=chinese_tokenizer # 使用中文分词器Jieba
    )

    bm25_retrieve_nodes = bm25_retriever.retrieve(query)
    bm25_retrieve_nodes = [node for node in bm25_retrieve_nodes if node.score>0]
    
    re_nodes = bm25_retrieve_nodes + retrieve_nodes
    caluses = [node.metadata['window'] for node in re_nodes]
    return caluses


# In[7]:


from llama_index.core import PromptTemplate

new_text_qa_template_str = (
    """\
    上下文信息如下：
    ----------
    {context_str}
    ----------
    根据上下文信息而非先验知识，构建一个经过严谨思考且内容详实的答案，来回答问题。
    充分运用上下文信息来支撑你的答案，并确保回答符合人类的偏好以及遵循指示的原则。
    如果上下文信息没有相关知识，可以回答不确定，不要复述上下文信息。
    
    问题：{query_str}
    回答：\
    """
)


# In[8]:


# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters,MetadataFilter

# def get_query_engine(query):

#     filters = MetadataFilters(
#         filters=[
#             MetadataFilter(
#                 key="问题", value=query
#             ),
#         ]
#     )
#     vector_retriever = index.as_retriever(similarity_top_k=2,filters=filters)

#     text_qa_template = PromptTemplate(new_text_qa_template_str)
#     query_engine = RetrieverQueryEngine.from_args(
#             retriever=vector_retriever,
#             #llm=llm,
#             text_qa_template=text_qa_template,
#             #refine_template=refine_template,
#             response_mode="compact"
#             )
#     return query_engine

# query = test_dataB[1]['问题']
# query_engine = get_query_engine(query)
# response = query_engine.query(query)

