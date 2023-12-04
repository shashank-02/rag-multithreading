from langchain.vectorstores.qdrant import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

#Loading the PDF and Splitting the text
loader = PyPDFLoader("javathread.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

print("PDF Successfully Loaded..!!")

#Loading the Embedding Model
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device' : 'cpu'}
encode_kwargs = {'normalize_embeddings' : False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

print("Model Successfully Loaded..!!")

#Port URL
url = "http://localhost:6333"


#Adding vectors to the Vector Database
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc = False,
    collection_name = "javathreads_db"
)
print("Java Threads - Vector Database Successfully Created..!!")