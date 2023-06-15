from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.chains import RetrievalQAWithSourcesChain

import os
import nltk
import config
import logging

# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

# Load documents from the specified directory using a DirectoryLoader object
loader = DirectoryLoader(config.FILE_DIR, glob='*.pdf')
documents = loader.load()

# split the text to chuncks of of size 1000
text_splitter = CharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.overlap)
# Split the documents into chunks of size 1000 using a CharacterTextSplitter object
texts = text_splitter.split_documents(documents)

# Create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])

# Define answer generation function
def answer(prompt: str, persist_directory: str = config.PERSIST_DIR) -> str:
    
    # Log a message indicating that the function has started
    LOGGER.info(f"Start answering based on prompt: {prompt}.")
    
    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["summaries", "question"])
    
    # Load a QA chain using an OpenAI object, a chain type, and a prompt template.

    doc_chain = load_qa_with_sources_chain(
        llm=OpenAI(
            openai_api_key = config.OPENAI_API_KEY,
            model_name="text-davinci-003",
            temperature=0,
            # max_tokens=config.max_tokens,
        ),
        chain_type=config.chain_type,
        prompt=prompt_template,
    )
    
    # Log a message indicating the number of chunks to be considered when answering the user's query.
    LOGGER.info(f"The top {config.k} chunks are considered to answer the user's query.")
    
    # Create a VectorDBQA object using a vector store, a QA chain, and a number of chunks to consider.
    # qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=config.k)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": config.k})
    qa = RetrievalQAWithSourcesChain(combine_documents_chain=doc_chain, retriever=retriever, return_source_documents=True)
    
    # Call the VectorDBQA object to generate an answer to the prompt.
    result = qa({"question": prompt})
    answer = result["answer"]
    
    # Log a message indicating the answer that was generated
    LOGGER.info(f"The returned answer is: {answer}")
    
    # Log a message indicating that the function has finished and return the answer.
    LOGGER.info(f"Answering module over.")
    return answer, result["source_documents"]
