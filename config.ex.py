OPENAI_API_KEY = "YOUR_API_KEY"  # replace with your actual OpenAI API key
PERSIST_DIR = "vectorstore"  # replace with the directory where you want to store the vectorstore
LOGS_FILE = "logs/log.log"  # replace with the path where you want to store the log file
FILE ="doc/CV.pdf" # replace with the path where you have your documents
FILE_DIR = "doc/"
"""
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""
prompt_template = """You are a personal Bot assistant for answering any questions about some provided documents.
You are given a question and a set of documents.
If the user's question requires you to provide specific information from the documents, give your answer based only on the information provided below. DON'T generate an answer that is NOT written in the provided examples.
If you do not know the answer, explicitly say that you do not know, do not try to make up an answer.

QUESTION: {question}

DOCUMENTS:
=========
{summaries}
=========
"""
k = 4  # number of chunks to consider when generating answer
chunk_size = 1000  # size of chunks to split documents into
overlap = 0  # size of overlap between chunks
max_tokens = 300 # maximum tokens in prompt

"""
Options:
- "stuff"
- "map_reduce"
"""
chain_type = "stuff"