import pandas as pd
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langsmith import traceable
from typing import List, Dict, Any
from langchain_mistralai import ChatMistralAI
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class MovieRAGWithDeepSearch:
    def __init__(
        self,
        csv_path: str,
        embedding_model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
        vector_database_path: str = "data/movie_chroma_db",
        chunk_size: int = 500,
        initial_k: int = 20,
        final_k: int = 8, 
        device: str = "cpu"
    ):
        """
        Deep Movie RAG with Multi-Stage Retrieval
        """
        self.csv_path = csv_path
        self.embedding_model_path = embedding_model_path
        self.llm_model_path = llm_model_path
        self.vector_database_path = vector_database_path
        self.chunk_size = chunk_size
        self.initial_k = initial_k
        self.final_k = final_k
        self.device = device

        self.df = None
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None

        self._load_and_prepare_data()
        self._setup_embeddings()
        self._build_vectorstore()
        self._setup_llm()
        self._setup_deep_qa_chain()

    def _load_and_prepare_data(self):
        """Load and preprocess TMDB CSV"""
        df = pd.read_csv(self.csv_path)
        required = ['title', 'overview']

        df = df.dropna() #subset=required)
        df = df[df['overview'].str.len() > 30]
        df = df.reset_index(drop=True)
        print(f"Loaded {len(df):,} movies after cleaning")

        self.df = df

    def _setup_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': self.device}
        )

    def _create_document_text(self, row) -> str:
        """Additional context for deep understanding"""

        parts = [
            f"Title: {row['title']}",
            f"Plot: {row['overview']}"
        ]

        if 'genres' in row and pd.notna(row['genres']):
            parts.append(f"Genres: {row['genres']}")
        if 'release_date' in row and pd.notna(row['release_date']):
            year = str(row['release_date'])[:4]
            parts.append(f"Year: {year}")
        if 'vote_average' in row and pd.notna(row['vote_average']):
            parts.append(f"Rating: {row['vote_average']:.1f}/10")
        if 'keywords' in row and pd.notna(row['keywords']):
            parts.append(f"Keywords: {row['keywords']}")

        return "\n".join(parts)

    def _create_documents(self) -> List[Document]:
        docs = []
        for _, row in self.df.iterrows():
            text = self._create_document_text(row)
            metadata = {
                'title': str(row['title']),
                'source': 'TMDB',
                'year': str(row.get('release_date', ''))[:4],
                'rating': float(row['vote_average']) if pd.notna(row.get('vote_average')) else 0.0,
                'genres': str(row.get('genres', '')),
                'tmdb_id': int(row['id']) if 'id' in row else -1
            }
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def _build_vectorstore(self):
        if os.path.exists(self.vector_database_path):
            print(f"Loading existing vector database from {self.vector_database_path}...")
            self.vectorstore = Chroma(
                persist_directory=self.vector_database_path,
                embedding_function=self.embeddings
            )
            return
        
        documents = self._create_documents()
        total = len(documents)
        print(f"Indexing {total} documents in batches of {self.chunk_size}...")

        self.vectorstore = Chroma.from_documents(
            documents=documents[:self.chunk_size],
            embedding=self.embeddings,
            persist_directory=self.vector_database_path
        )

        for i in range(self.chunk_size, total, self.chunk_size):
            batch = documents[i:i + self.chunk_size]
            self.vectorstore.add_documents(batch)

        self.vectorstore.persist()
        print(f"Vector DB saved to {self.vector_database_path}")

    def _setup_llm(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env file!")

        print("Loading Mistral-7B-Instruct...")
        # tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
        # model = AutoModelForCausalLM.from_pretrained(
        #     self.llm_model_path,
        #     device_map=self.device,
        # )
        # pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     max_new_tokens=512,
        #     temperature=0.0,
        #     return_full_text=False,  
        #     do_sample=False, 
        #     eos_token_id=tokenizer.eos_token_id,
        #     pad_token_id=tokenizer.pad_token_id,
        # )
        # self.llm = HuggingFacePipeline(pipeline=pipe)
        self.llm = ChatMistralAI(
            model="open-mistral-7b",
            temperature=0,
            max_retries=2,
            api_key=api_key,
        )

    def _setup_deep_qa_chain(self):
        """Multi-stage: Retrieve → Rerank → Generate"""
        template = """You are a precise movie expert. Use ONLY the provided movie information to answer.

            MOVIES (Context):
            {context}
            
            USER QUESTION: {question}
            
            INSTRUCTIONS:
            1. Analyze the provided text to find the ONE movie that best answers the specific user question.
            2. Select exactly 1 movie.
            3. Return ONLY the exact title of the movie.
            4. Do not write introductions, numbering, or explanations. Just the title string.
        
        RESPONSE:"""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.initial_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def search_movies(self, query: str, k: int = 25) -> List[Dict]:
        """Raw semantic search"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                'title': d.metadata['title'],
                'year': d.metadata['year'],
                'rating': d.metadata['rating'],
                'genres': d.metadata['genres'],
                'tmdb_id': d.metadata['tmdb_id']
            }
            for d in docs
        ]

    def recommend(self, query: str) -> Dict[str, Any]:
        """Deep RAG: Retrieve → Generate"""
        result = self.qa_chain({"query": query})
        answer = result["result"].strip()

        lines = [line.strip() for line in answer.split('\n') if line.strip() and not line.startswith('-')]
        predicted_titles = [line.split('. ')[-1] if '. ' in line else line for line in lines]

        return {
            'answer': answer,
            'predicted_titles': predicted_titles,
            'source_documents': [
                {
                    'title': doc.metadata['title'],
                    'tmdb_id': doc.metadata['tmdb_id'],
                    'content': doc.page_content[:300]
                }
                for doc in result['source_documents'][:self.final_k]
            ],
            'retrieved_count': len(result['source_documents'])
        }

    def recommend_formatted(self, query: str) -> str:
        result = self.recommend(query)
        output = f"QUERY: {query}\n\n"
        output += "Answer:" + "\n\n"

        description = ""
        for i, doc in enumerate(result['source_documents'], 1):
            if doc['title'] == result['answer']:
                description = doc['content']
                break

        output += "Based on your description, the most suitable movie is " + description[6:] + ".\n"
        return output

    def chat(self, query: str, use_rag: bool = True) -> str:
        if not use_rag:
            response = self.llm(f"You are a movie expert. {query}")
            return f"Answer: {response}"
        return self.recommend_formatted(query)

    @traceable
    def rag_function(self, inputs: dict) -> dict:
        """Function for evaluation with LangSmith"""
        query = inputs["query"]
        result = self.recommend(query)
        return {
            "answer": result["answer"],
            "retrieved_docs": [
                {"id": doc["tmdb_id"], "title": doc["title"]}
                for doc in result["source_documents"]
            ],
            "predicted_titles": result["predicted_titles"]
        }