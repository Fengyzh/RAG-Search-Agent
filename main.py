from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.utilities import SearxSearchWrapper
import ollama
import PyPDF2
import pprint




config = {"model":"mistral",
          "local_RAG": True, 
          "dir_path": "./RAG_dir", 
          "init_embedding":True,
          "embedding_dir":"./embedding", 
          "search": True}

def local_RAG():
    if config["local_RAG"]: 
        path = config["dir_path"]
        #text_loader_kwargs={'encoding': "UTF-8"}
        loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        #print(docs[0].page_content)
        #print(docs[0].metadata)
        #embeddings = OllamaEmbeddings(model=config["model"])
        embeddings = HuggingFaceBgeEmbeddings(
            model_name = 'BAAI/bge-large-en-v1.5',
            model_kwargs = {"device": "cuda"},
            encode_kwargs = {"normalize_embeddings": True}
        )
        if config["init_embedding"]:
            print("----- Initial embedding enabled -----")
            text_spliter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=0)
            splits = text_spliter.split_documents(docs)
            vectorstores = Chroma.from_documents(splits, embedding=embeddings, persist_directory=config["embedding_dir"])
        else:
            vectorstores = Chroma(persist_directory=config["embedding_dir"], embedding_function=embeddings)
        

        v_retriever = vectorstores.as_retriever(search_kwargs={'k':5})
        bm25_retriever = BM25Retriever.from_documents(splits, kwargs=5)
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, v_retriever], weights=[0.5, 0.5]
        )
        return retriever


def format_docs(docs):
    print(docs[0].metadata)
    doc_page = ""
    for doc in docs:
        doc_page += doc.page_content
        doc_page += "\t"
        Source_con = "Source: " + doc.metadata["source"]
        doc_page += Source_con
        doc_page += "\n\n"
    return doc_page


def ollama_llm(q, c):
    formatted_prompt = f"Please answer the question in detail using the context provided. Use the context provided for the reponse only! Source will be provided in the context, please state the source in each reference. Question: {q}\n\nContext: {c}"
    print("formatted prompt: ", formatted_prompt)
    response = ollama.chat(
    model='dolphin-mistral',
    messages=[{'role': 'user', 'content': formatted_prompt}],
    stream=True)
    return response


def rag_chain(q):
    retriever = local_RAG()
    retrieved_docs = retriever.invoke(q)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(q, formatted_context)  

#result = rag_chain("What is the script about? and write the documention according to the steps about a Python coding project")

#print("--------------------- AI RESPONSE ----------------------------")
#for chunk in result:
#    print(chunk['message']['content'], end='', flush=True)

def q_web_format(web):
    #return web.replace("\n", "")
    temp = web.split("\\n")
    doc = "".join(temp)
    return doc.replace("\n", " ")


def internet_search(query):
    urls = []
    search = SearxSearchWrapper(searx_host="http://localhost:8080")
    results = search.results(
        query,
        num_results=1,
        engines="google",
        time_range="year",
    )
    #pprint.pp(results[0]['link'])
    for j in range(len(results)):
        urls.append(results[j]['link'])
    url_loader = SeleniumURLLoader(urls=urls)
    data = url_loader.load()

    for i in data:
        print("Content Sources: ", i.metadata['source'])
    print("--- Source Link Above ---")

    for i in range(len(urls)):
        print(f"\n\n ------- No.{i+1} Doc ------- \n\n")
        source = data[i].metadata['source']
        #pprint.pp(q_web_format(data[0].page_content))

        #prompt = f"In great detail can you list and summarize the following website content? Use the provided content only! include a conclusion as well. Source link will be provided in the 'Source: ' you must state the source link at the beginning of your response with the format 'Source: (The Link Provided)' Only. \n Source:{source} Content: \n\n{q_web_format(data[i].page_content)} \n\n"

        prompt = f"List the following content, you must use the content provided only! If you receive a error response code based on the content, you must state 'Fail to load website' and stop. Source link will be provided in the 'Source: ' you must state the source link at the beginning of your response. \n Source:{source} Content: \n\n{q_web_format(data[0].page_content)} \n\n"

        print(q_web_format(data[0].page_content))
        print("----- Reponse -----")

        response = ollama.chat(
        model='dolphin-mistral',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True)

        for chunk in response:
            print(chunk['message']['content'], end='', flush=True)


#internet_search("-site:wikipedia.org Why are entry level jobs require so many experience")
p = "what is the paper about"
result = rag_chain(p)

print("--------------------- AI RESPONSE ----------------------------")
for chunk in result:
    print(chunk['message']['content'], end='', flush=True)
#internet_search(p)



def read_pdf(file_path):
    # Open the PDF file in binary mode
    with open(file_path, 'rb') as file:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Get the total number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        
        # Initialize an empty string to store the text
        text = ""
        
        # Iterate through each page of the PDF
        for page_number in range(num_pages):
            # Get the specified page
            page = pdf_reader.pages[page_number]
            
            # Extract text from the page
            page_text = page.extract_text()
            # Append the text of the current page to the overall text
            text += page_text
            
        return text

# Example usage:
#pdf_file_path = './RAG_dir/Final 2019LE-CS Technical Paper.pdf'  # Replace 'example.pdf' with the path to your PDF file
#pdf_text = read_pdf(pdf_file_path)
#print(pdf_text)



