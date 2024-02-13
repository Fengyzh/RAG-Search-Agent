from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import ollama
import PyPDF2



config = {"model":"mistral",
          "local_RAG": True, 
          "dir_path": "./RAG_dir", 
          "init_embedding":False,
          "embedding_dir":"./embedding", 
          "search": True}

if config["local_RAG"]: 
    path = config["dir_path"]
    text_loader_kwargs={'encoding': "UTF-8"}
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    #print(docs[0].page_content)
    #print(docs[0].metadata)
    embeddings = OllamaEmbeddings(model=config["model"])
    if config["init_embedding"]:
        print("----- Initial embedding enabled -----")
        text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        splits = text_spliter.split_documents(docs)
        vectorstores = Chroma.from_documents(splits, embedding=embeddings, persist_directory=config["embedding_dir"])
    else:
        vectorstores = Chroma(persist_directory=config["embedding_dir"], embedding_function=embeddings)
    
    retriever = vectorstores.as_retriever(search_kwargs={'k':5})


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
    formatted_prompt = f"Please answer the question in detail using the context provided, Source will be provided in the context, please state the source in each reference. Question: {q}\n\nContext: {c}"
    print("formatted prompt: ", formatted_prompt)
    response = ollama.chat(
    model='mistral',
    messages=[{'role': 'user', 'content': formatted_prompt}],
    stream=True)
    return response


def rag_chain(q):
    retrieved_docs = retriever.invoke(q)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(q, formatted_context)  

result = rag_chain("What is the script about and write the documention according to the steps about a Python coding project?")

print("--------------------- AI RESPONSE ----------------------------")
for chunk in result:
    print(chunk['message']['content'], end='', flush=True)







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
pdf_file_path = 'Final 2019LE-CS Technical Paper.pdf'  # Replace 'example.pdf' with the path to your PDF file
#pdf_text = read_pdf(pdf_file_path)
#print(pdf_text)



