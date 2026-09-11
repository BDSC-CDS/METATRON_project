import pandas as pd
from pymilvus import MilvusClient, Collection, connections
from collections import Counter
from llama_index.embeddings.openai import OpenAIEmbedding
from models import *

embed_model = OpenAIEmbedding(model_name=models["embedder"]["model_name"], api_base=models["embedder"]["url"], embed_batch_size=1, api_key="empty")


# Explore collection
# Pragraphs 
reference_path = "../data/"
paragraph_tb = pd.read_csv(reference_path + "paragraphs_tb.csv")
print(f"Paragraphs len: {len(paragraph_tb)}")

collection_name = "metatron_collection"
connections.connect(uri=milvus_uri)
collection = Collection(collection_name)
print(f"Num. entities: {collection.num_entities}")

# Create index 
index_params = {
    "index_type": "FLAT",   
    "metric_type": "COSINE",    
    "params": {}    
}
collection.create_index(
    field_name="embedding",
    index_params=index_params
)

# Load collection
collection.load() 
# Explore topic contribution 
topic_list = ["PrimaryTumor", "LiverMetastases", "RectalPrimaryTumor", "ExtraHepaticMetastases"]
for t in topic_list:
    res = collection.query(
        expr=f'topic == "{t}"',
        output_fields=["id"], 
        limit=collection.num_entities        
    )
    count = len(res)
    print(f"{t}: {count} ({count/collection.num_entities})")

# Try filter 
def search_on_db(query: str):
    search_res = collection.search(
        data=[embed_model.get_text_embedding(query)], 
        limit=30,  # Return top 15 results
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": collection.num_entities}}, 
        output_fields=["id", "topic"], 
        expr='topic in ["RectalPrimaryTumor"]'
    )
    return search_res