from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document  # Importing Document schema from Langchain
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

import tiktoken  # token counter (Compatible with OpenAI Embeddings model)

import os, sys, shutil

# logging setup
import logging
from logging.handlers import RotatingFileHandler
from logging.config import dictConfig

REQUIRED_ENV_VARS = {"OPENAI_API_KEY"}
CHROMA_PATH = "./chroma"


class RagChain:
    """Initialise a Rag Chabot instance with built in logging. Authenticate via a KEY set as an OS env var.

    """
    def __init__(self):
        # check required environment variables are set
        diff = REQUIRED_ENV_VARS.difference(os.environ)
        if len(diff) > 0:
            raise EnvironmentError(f'Failed because {diff} environment variables not set')

        self.template = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # test if LOG_DIR set, else LOG_DIR = ./logs
        log_dir = os.getenv("LOG_DIR")
        if not log_dir:
            self.log_dir = './logs'
        # test if LOG_LEVEL set, else LOG_LEVEL = 'INFO'
        log_level = os.getenv('LOG_LEVEL')
        if not log_level:
            self.log_level = 'INFO'

        # initialise logging
        self.logger = self._init_logging()

    def _init_logging(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        file_handler = RotatingFileHandler(self.log_dir + "/ragbot.log", maxBytes=10240, backupCount=10)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))

        dictConfig({
            'version': 1,
            'formatters': {'default': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            }},
        })

        logger = logging.getLogger('ragbot')
        logger.addHandler(file_handler)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M'))
        #logging.getLogger().addHandler(console)
        logger.addHandler(console)

        if self.log_level == 'DEBUG':
            file_handler.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        elif self.log_level == 'WARNING':
            file_handler.setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)
        else:
            file_handler.setLevel(logging.INFO)
            logger.setLevel(logging.INFO)

        logger.info('Initialized logging')
        return logger

    def load_template(self, template_file):
        """ load a prompt template from a local text file in folder templates/
        """
        file_path = "templates/" + template_file
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents = file.read()
        self.template = file_contents

    def load_pdf_docs(self, data_path):
        """ Load PDF documents from the specified path using PyPDFDirectoryLoader.
        Args:
            :data_path str: path to locate PDF docs in - use a relative path
        Returns:
            List of Langchain Document objects
        """
        self.logger.info("Loading PDF documents")

        # fix to supress "Ignoring wrong pointing object" warning messages
        logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

        # Initialize PDF loader with specified directory
        document_loader = PyPDFDirectoryLoader(data_path)
        documents = document_loader.load()

        self.logger.info(f"Loaded {len(documents)} PDF pages")
        return documents

    def split_text(self, documents: list[Document]):
        """Split the text content of the given list of Document objects into smaller chunks.
        Args:
            documents (list[Document]): List of Document objects containing text content to split.
        Returns:
            list[Document]: List of Document objects representing the split text chunks.
        """

        self.logger.info("Splitting text into chunks")
        # Initialize text splitter with specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Size of each chunk in characters
            chunk_overlap=100,  # Overlap between consecutive chunks
            length_function=len,  # Function to compute the length of the text
            add_start_index=True,  # Flag to add start index to each chunk
        )

        # Split documents into smaller chunks using text splitter
        return text_splitter.split_documents(documents)

    def save_embeddings_chroma(self, chunks: list[Document]):
        """
        Save the given list of Document objects to a Chroma database.
        Args:
            :chunks (list[Document]): List of Document objects representing text chunks to save.
        Returns:
            None
        """

        self.logger.info("Generating embeddings and saving text chunks and embeddings to vector database")
        # Clear out the existing database directory if it exists
        self.logger.info(f"ChromaDB database path: {CHROMA_PATH}")
        if os.path.exists(CHROMA_PATH):
            self.logger.info("Deleting previous ChromaDB database")
            shutil.rmtree(CHROMA_PATH)

        # generate and save embeddings for the chunks in ChromaDB to disk
        db = Chroma.from_documents(chunks, self.embeddings, persist_directory=CHROMA_PATH)

        all_docs = db.get(include=["documents", "embeddings"])
        self.logger.info(f"Saved {len(all_docs['embeddings'])} embeddings")
        self.logger.debug(f"Sample of vector database embedding {all_docs['embeddings'][0]}")

    def chroma_dump(self):
        # Open the persisted database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        # Retrieve all documents and metadata
        all_docs = db.get()

        print(all_docs.keys())
        print(all_docs['data'])
        # Print all stored documents and their metadata
        # for i, doc in enumerate(all_docs, start=1):
        #    print(f"Document {i}:")
        #    #print(f"Content: {doc['page_content']}")
        #    print(f"Metadata: {doc['metadata']}")

    def query_chroma(self, query_text, k_int=1, relevance_score=0.1):
        """
        Do a similarity search for a given text string in the ChromaDB database.
        Also count the tokens involved so we know how much information is processed token-by-token when sending this to LLM
        Langchain Chroma docs here:
         https://api.python.langchain.com/en/latest/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html
         "Lower score represents more similarity"
        Args:
            :query_text txt: - text to query Chroma for
            :k_int int: - number of results to return ("top-n")
            :relevance_score float: - threshold below which search results are discarded
        Returns:
            :results: - (List[Tuple[Document, float]] of ChromaDB search results, concat_context_txt, token_count)
        """
        self.logger.debug(f"Opening ChromaDB database at {CHROMA_PATH}")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)

        # Load the tokenizer for the model being used - cl100k_base = OpenAI model. Tokenizer is for counting, not search
        tokenizer = tiktoken.get_encoding("cl100k_base")  # Use the encoding for OpenAI models

        # helper function to count tokens
        def count_tokens(text):
            return len(tokenizer.encode(text))

        # Perform a similarity search to get results "r"
        r = db.similarity_search_with_relevance_scores(query_text, k=k_int)
        self.logger.debug(f"Returning {len(r)} ChromaDB search results")

        # Check if there are any matching results or if the relevance score is too low
        if len(r) == 0 or r[0][1] < relevance_score:
            self.logger.warning("No vector search results found with high enough relevance score")
            return (None, None, 0)
        else:
            # Combine context from matching documents
            context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in r])
            # how many tokens is when sending back to LLM to process as part of prompt?
            token_count = count_tokens(context_text)

            return (r, context_text, token_count)

    def query_rag(self, query_text, template):
        """
        Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
        Args:
          :query_text (str): The text to query the RAG system with.
          :template (str): prompt template to pass to the bot
        Returns:
          :formatted_response (str): Formatted response including the generated text and sources.
          :response_text (str): The generated response text.
        """

        # Retrieving the context from the DB using similarity search
        # results = db.similarity_search_with_relevance_scores(query_text, k=3)
        results, joined_text, tokens = self.query_chroma(query_text, 3)

        # Check if there are any matching results
        if results is None:
            return None, "No Data Found"

        # Combine context from matching documents
        context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

        # Create prompt template using context and query text
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Initialize OpenAI chat model
        model = ChatOpenAI()

        # Generate response text based on the prompt
        response_text = model.predict(prompt)

        # Get sources of the matching documents
        sources = [doc.metadata.get("source", None) for doc, _score in results]

        # Format and return response including generated text and sources
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return formatted_response, response_text



    def run_docs_load(self, data_path):
        """ Run a document load, split and save to vector operation """
        documents = self.load_pdf_docs(data_path)
        chunks = self.split_text(documents)
        self.save_embeddings_chroma(chunks)

    def run_query_db(self, query_string, k_int):
        """ Query the vector database"""
        results, joined_text, tokens = self.query_chroma(query_string, k_int)
        # display the joined text that would be passed to the LLM prompt as context
        print(joined_text)

        # Display the results metadata
        for i, result in enumerate(results, start=1):
            doc, sim = result
            print(f"{i}: score:{round(sim,3)}, page:{doc.metadata['page']}, doc:{doc.metadata['source']}, index:{doc.metadata['start_index']}")
        print(f"Token Count {tokens}")

    def run_rag_chat(self, query_string, template="simple_rag_qanda.txt"):
        """ RAG Query Chat"""
        self.load_template(template)
        # rag_qanda_here
        formatted_response, response_text = self.query_rag(query_string, self.template)
        print(response_text)

    def run_llm_chat(self, query_string):
        """ Chat directly with the LLM, no context"""
        # Initialize OpenAI chat model
        model = ChatOpenAI()

        # Generate response text
        response_text = model.predict(query_string)
        print(response_text)







