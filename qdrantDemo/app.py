from langchain.vectorstores.qdrant import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

#Initializing Model
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device' : 'cpu'}
encode_kwargs = {'normalize_embeddings' : False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

url = "http://localhost:6333"

client = QdrantClient(
    url = url,
    prefer_grpc = False
)

#Test - 1
#print("Qdrant Client")
#print(client)

db = Qdrant(client=client, embeddings=embeddings, collection_name="javathreads_db")
#print("Qdrant Database")
#print(db)

query = str(input("State your question on multithreading concept in Java....    "))

#Test - 2
#result = db.similarity_search_with_score(query = query, k=1)
#print(result)

#Fetching result from Vector Database for the query
infos = db.similarity_search_with_score(query=query, k=1)
for i in infos:
    info, score = i
    print(info.page_content) 

