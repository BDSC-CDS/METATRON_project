import os 
import pandas as pd 
import numpy as np
from pymilvus import MilvusClient, connections, Collection, CollectionSchema, FieldSchema, DataType
from llama_index.embeddings.openai import OpenAIEmbedding
from tqdm import tqdm 
import collections
from models import models, milvus_uri

# FUNCTIONS
# Get paragraphs embeddings 
def embed_text(text: str):
    return embed_model.get_text_embedding(text)

# Pragraphs 
reference_path = "../data/"
paragraph_tb = pd.read_csv(reference_path + "paragraphs_tb.csv")


'''
# If not already done.....
# Check and remove duplicated paragraphs
content_list = paragraph_tb["Content"].to_list()
unique_content = [item for item, count in collections.Counter(content_list).items() if count == 1]
multiple_content = [item for item, count in collections.Counter(content_list).items() if count > 1]
print(f"Before -- Tb len: {len(paragraph_tb)}; Unique content: {len(unique_content)}; Repeated content: {len(multiple_content)}")

paragraph_tb = paragraph_tb.drop_duplicates(subset=["Content"], keep="first").reset_index()
content_list = paragraph_tb["Content"].to_list()
unique_content = [item for item, count in collections.Counter(content_list).items() if count == 1]
multiple_content = [item for item, count in collections.Counter(content_list).items() if count > 1]
print(f"After -- Tb len: {len(paragraph_tb)}; Unique content: {len(unique_content)}; Repeated content: {len(multiple_content)}")

print(f"# References: {len(paragraph_tb)}")
print(f"# Included studies: {len(paragraph_tb.loc[paragraph_tb["Paragraph_N"]==1])}")
# paragraph_tb.to_csv("studies/paragraphs_tb_final.csv",index=False)
'''

# Embedding model
embed_model = OpenAIEmbedding(model_name=models["embedder"]["model_name"], api_base=models["embedder"]["url"], embed_batch_size=1, api_key="empty")
# Milvus db 
print(f"Milvus DB path: {milvus_uri}")
milvus_client = MilvusClient(milvus_uri)
collection_name = "metatron_collection"

# # Check if previous collection was created 
if milvus_client.has_collection(collection_name):
    connections.connect(uri=milvus_uri)
    coll = Collection(collection_name)
    print(f"# Entities (before): {coll.num_entities}")

# Clean collection
if milvus_client.has_collection(collection_name):
    print("Previous collection will be deleted...")
    milvus_client.drop_collection(collection_name)

# Create new collection
# Create Schema 
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    # FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=512)
]
schema = CollectionSchema(fields=fields, description="RAG document store")
# Create collection with schema 
if not(milvus_client.has_collection(collection_name)):
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema
        #dimension=1024
    )
print(f"Collection exists: {milvus_client.has_collection(collection_name)}")
# If yes
if milvus_client.has_collection(collection_name):
    # Check collection num entities
    connections.connect(uri=milvus_uri)
    coll = Collection(collection_name)
    print(f"# Entities (before): {coll.num_entities}")

    # Populate collection
    data = []
    for i, row in enumerate(tqdm(paragraph_tb.itertuples(index=False), desc="Creating embeddings", total=len(paragraph_tb))):
        id = row.Reference_ID
        content = row.Content
        author = row.Author
        title = row.Title
        topic = row.Topic
        embedding = embed_text(content)
        data.append({"id": id, "embedding": embedding, "author":author, "title":title, "topic":topic})
    
    coll.insert(data)   
