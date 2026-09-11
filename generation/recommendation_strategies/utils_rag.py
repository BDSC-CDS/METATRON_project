import pandas as pd 
import json
from pymilvus import MilvusClient, Collection, connections
from llama_index.embeddings.openai import OpenAIEmbedding
from models import *

data_path = "../data/"

# Settings 
# References data 
paragraphs_tb = pd.read_csv(data_path + "paragraphs_tb.csv")
# Embedding model
embed_model = OpenAIEmbedding(model_name=models["embedder"]["model_name"], api_base=models["embedder"]["url"], embed_batch_size=1, api_key="empty")
# Milvus db collection
collection_name = "metatron_collection"
connections.connect(uri=milvus_uri)
collection = Collection(collection_name)

def embed_text(text: str):
    return embed_model.get_text_embedding(text)

def implement_filter(case_n: int, pt_information= pd.read_csv(data_path + "sie_Colorectal_Primary_Tumor_Characterization.csv"),
                     conditions_information=pd.read_csv(data_path + "conditional_steps_results.csv"), 
                     ehm_fields=["Extrahepatic_Pulmonary_Involvement","Extrahepatic_Peritoneal_Involvement","Extrahepatic_Lymph_Node_Involvement","Extrahepatic_Bone_Involvement", "Extrahepatic_Brain_Involvement"]):
    # First component: PrimaryTumor
    row = pt_information.loc[pt_information["Case_ID"]==case_n,:]
    pt_location = row.Location.item()
    pt = "RectalPrimaryTumor" if "rect" in pt_location else "PrimaryTumor"
    # Third component: Extrahepatic metastases
    ehm_flag = False
    for f in ehm_fields:
        if conditions_information.loc[(conditions_information["Substep"]==f) & (conditions_information["Case"]==case_n), "Verified"].item() == "True":
            ehm_flag = True
            break
    filter_list = [f"{pt}", "LiverMetastases"]
    if ehm_flag:
        filter_list.append("ExtraHepaticMetastases")
    return filter_list


def search_on_db(query: str, filter: str, n_res: int):
    if filter != "":
        search_res = collection.search(
            data=[embed_model.get_text_embedding(query)], 
            limit=n_res,  
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": collection.num_entities}}, 
            output_fields=["id", "title", "author", "topic"], 
            expr=filter
        )
    else:
        search_res = collection.search(
            data=[embed_model.get_text_embedding(query)], 
            limit=n_res, 
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": collection.num_entities}}, 
            output_fields=["id", "title", "author", "topic"]
        )        
    return search_res

def build_detailed_excerpts(query: str, filter: str, n_res: int, paragraphs_tb= paragraphs_tb):
    search_res = search_on_db(query, filter, n_res)
    refs, sims = [], []
    if len(search_res[0]) > 0:
        for s in search_res[0]:
            id = s["entity"]["id"]
            author = s["entity"]["author"]
            title = s["entity"]["title"]
            row = paragraphs_tb.loc[paragraphs_tb["Reference_ID"]==id,:]
            content = row["Content"].item()
            refs.append({"Reference content":content.replace("\\n","\n"), "Reference author":author, "Reference title":title})
            sims.append({"Reference id":id, "Sim":s["distance"]})
    return refs, sims

def build_excerpts(query: str, filter: str, n_res: int, paragraphs_tb= paragraphs_tb):
    search_res = search_on_db(query, filter, n_res)
    refs, sims = [], []
    if len(search_res[0]) > 0:
        ex_counter = 1
        for s in search_res[0]:
            id = s["entity"]["id"]
            row = paragraphs_tb.loc[paragraphs_tb["Reference_ID"]==id,:]
            content = row["Content"].item()
            refs.append(content.replace("\\n","\n"))
            sims.append({"Reference id":id, "Sim":s["distance"]})
            ex_counter += 1
    return refs, sims

def retrieval_information(query: str, filter: str, n_res: int):
    search_res = search_on_db(query, filter, n_res)
    refs = []
    if len(search_res[0]) > 0:
        for s in search_res[0]:
            title = s["entity"]["title"]
            topic = s["entity"]["topic"]
            refs.append({"Title":title, "Topic":topic})
    return refs