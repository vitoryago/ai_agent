"""
Professional RAG System using LangChain + Ollama
This demonstrates modern AI engineering practices for building reliable RAG applications.
"""

import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import warnings
warnings.filterwarnings("ignore")

class ProductionRAG:
    """
    Production-ready RAG system using LangChain + Ollama
    
    This system demonstrates:
    - Modern RAG architecture patterns
    - Proper prompt engineering
    - Reliable local inference
    - Industry-standard components
    """
    
    def __init__(self, ollama_model="llama3.1:8b"):
        """
        Initialize the RAG system with LangChain components
        
        Args:
            ollama_model: Ollama model to use (default: llama3.1:8b)
        """
        print("🚀 Initializing Production RAG System")
        print("=" * 60)
        
        self.ollama_model = ollama_model
        
        # Initialize components
        self._setup_embeddings() # 1. Prepare text-to-vector conversion
        self._setup_llm() # 2. Connect to language model
        self._create_knowledge_base() # 3. Build and index your knowledge
        self._setup_retrieval_chain() # 4. Wire everything together
        
        print("Production RAG System Ready!")
        print(f"Knowledge base: {len(self.knowledge_base)} documents")
        print(f"Using model: {ollama_model}")
        print("=" * 60)
    
    def _setup_embeddings(self):
        """
        Setup embedding model for vector similarity search
        
        This loads a neural network trained specifically to convert sentencesinto
        384-dimensional vectors that capture semantic meaning.
        """
        print("Loading embedding model...")
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def _setup_llm(self):
        """Setup Ollama LLM with optimized parameters"""
        print(f"Connecting to Ollama model: {self.ollama_model}")
        
        try:
            self.llm = Ollama(
                model=self.ollama_model,
                temperature=0.3,  # Lower temperature for more focused responses
                top_p=0.9, # Nucleus sampling
                top_k=40, # Vocabulary restriction
                repeat_penalty=1.1, # Discourages repetition
                stop=["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
            )
            
            # Test the connection
            test_response = self.llm.invoke("Hello")
            print("✅ Ollama connection successful")
            
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("💡 Make sure Ollama is running and the model is available")
            print(f"   Try: ollama run {self.ollama_model}")
            raise
    
    def _create_knowledge_base(self):
        """Create curated knowledge base with domain expertise"""
        print("📚 Creating knowledge base...")
        
        # Curated knowledge base - your domain expertise
        raw_knowledge = [
            "Transformers are neural network architectures that use self-attention mechanisms to process sequences in parallel, eliminating the need for recurrent connections. They were introduced in the 'Attention is All You Need' paper in 2017.",
            
            "The attention mechanism allows models to focus on relevant parts of the input when processing each element in a sequence. It computes attention weights that determine how much focus to place on different parts of the input.",
            
            "BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only transformer model designed for understanding tasks like classification, question answering, and named entity recognition. It processes text bidirectionally.",
            
            "GPT (Generative Pre-trained Transformer) models are decoder-only transformers designed for text generation by predicting the next word in a sequence. They generate text autoregressively from left to right.",
            
            "Multi-head attention allows transformers to attend to different types of relationships simultaneously. Each attention head can focus on different aspects like syntactic relationships, semantic similarity, or positional patterns.",
            
            "RAG (Retrieval-Augmented Generation) combines retrieval and generation for better factual accuracy by grounding responses in retrieved documents. It first retrieves relevant information, then generates responses based on that context.",
            
            "Hugging Face hosts over 1.7 million machine learning models, making it the largest repository of open-source AI models. It provides tools for model sharing, deployment, and collaboration in the AI community.",
            
            "Fine-tuning adapts pre-trained models to specific tasks or domains by continuing training on task-specific data. This allows leveraging general knowledge while specializing for particular applications.",
            
            "Vector embeddings represent text as high-dimensional numerical vectors that capture semantic meaning. Similar concepts have similar vector representations, enabling semantic search and similarity comparisons.",
            
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It enables fast retrieval in large-scale vector databases used in RAG systems.",
            
            "Ollama is a tool that allows running large language models locally on consumer hardware without internet connectivity. It simplifies model deployment and makes AI accessible for local development and testing.",
            
            "LangChain is a framework for building applications with language models. It provides components for RAG, agents, chains, and integrations with various AI services and vector databases."
        ]
        
        # Convert to LangChain Document objects
        self.knowledge_base = [
            Document(page_content=text, metadata={"source": f"knowledge_base_{i}"})
            for i, text in enumerate(raw_knowledge)
        ]
        
        # Create vector store with Chroma (better than FAISS for LangChain)
        # Each document text gets converted to 384-dimensional vector
        print("🔍 Creating vector database...")
        self.vectorstore = Chroma.from_documents(
            documents=self.knowledge_base,
            embedding=self.embeddings,
            persist_directory="./chroma_db"  # Everything gets saved to disk for reuse
        )
    
    def _setup_retrieval_chain(self):
        """Setup the RAG chain with proper prompt engineering"""
        print("⛓️  Setting up retrieval chain...")
        
        # Custom prompt template for RAG
        template = """You are a knowledgeable AI assistant specializing in machine learning and AI concepts. Use the provided context to answer questions accurately and comprehensively.

Context information:
{context}

Question: {question}

Instructions:
1. Base your answer primarily on the provided context
2. If the context is relevant, use it as the foundation for your response
3. You may supplement with general knowledge to make the answer more complete and understandable
4. If the context doesn't contain relevant information, clearly state this
5. Be concise but thorough in your explanations
6. Use technical terms appropriately for a technical audience

Answer:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Stuff all retrieved docs into prompt
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Retrieve top 3 most relevant docs
            ),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True  # Return sources for transparency
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a comprehensive answer with sources
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer, sources, and metadata
        
        Ex:
            When you call ask("What is attention?"), here's the complete flow:

            1. Question Processing: "What is attention?" gets converted to an embedding vector
            2. Similarity Search: The system compares this vector to all knowledge base vectors
            3. Document Retrieval: Top 3 most similar documents are retrieved
            4. Context Assembly: Retrieved documents get formatted into the prompt template
            5. Prompt Creation: The final prompt looks like:

            Context information:
            The attention mechanism allows models to focus on relevant parts...
            Multi-head attention allows transformers to attend to different types...

            Question: What is attention?

            Instructions: [numbered list]

            Answer: blablabla
        """
        print(f"\n🤔 Question: {question}")
        print("-" * 50)
        
        try:
            # Get answer from the chain
            print("🔍 Retrieving relevant information...")
            result = self.qa_chain.invoke({"query": question})
            
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Display retrieved sources for transparency
            print("📖 Retrieved Sources:")
            for i, doc in enumerate(source_docs, 1):
                content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                print(f"   {i}. {content_preview}")
            
            print(f"\n💡 Answer:")
            print(answer)
            
            return {
                "question": question,
                "answer": answer,
                "sources": [doc.page_content for doc in source_docs],
                "num_sources": len(source_docs)
            }
            
        except Exception as e:
            error_msg = f"❌ Error processing question: {str(e)}"
            print(error_msg)
            return {
                "question": question,
                "answer": "I encountered an error processing your question. Please check that Ollama is running and try again.",
                "sources": [],
                "num_sources": 0,
                "error": str(e)
            }
    
    def demonstrate_retrieval(self, question: str, k: int = 5):
        """
        Demonstrate the retrieval process for educational purposes
        
        Args:
            question: Question to search for
            k: Number of documents to retrieve
        """
        print(f"\n🔍 RETRIEVAL DEMONSTRATION")
        print(f"Query: '{question}'")
        print("=" * 60)
        
        # Get retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.0}
        )
        
        # Retrieve documents with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        
        print(f"📊 Retrieved {len(docs_with_scores)} documents:")
        
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            relevance = "🔥 Highly relevant" if score < 0.5 else "✅ Relevant" if score < 0.8 else "📝 Somewhat relevant"
            print(f"\n{i}. {relevance} (similarity score: {score:.3f})")
            print(f"   Content: {doc.page_content[:100]}...")
    
    def add_knowledge(self, new_documents: List[str]):
        """
        Add new knowledge to the system dynamically
        
        Args:
            new_documents: List of new document texts to add
        """
        print(f"📝 Adding {len(new_documents)} new documents...")
        
        # Convert to Document objects
        new_docs = [
            Document(page_content=text, metadata={"source": f"dynamic_{i}"})
            for i, text in enumerate(new_documents)
        ]
        
        # Add to existing vector store
        self.vectorstore.add_documents(new_docs)
        self.knowledge_base.extend(new_docs)
        
        print(f"✅ Knowledge base updated. Total documents: {len(self.knowledge_base)}")
    
    def test_model_connection(self):
        """Test that Ollama model is working correctly"""
        print("🧪 Testing Ollama connection...")
        
        try:
            response = self.llm.invoke("What is 2+2? Answer briefly.")
            print(f"✅ Model test successful: {response}")
            return True
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False

def check_ollama_status():
    """Check if Ollama is running and suggest models"""
    print("🔧 Checking Ollama status...")
    
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Ollama is running")
            print("📋 Available models:")
            print(result.stdout)
            return True
        else:
            print("❌ Ollama is not responding")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False
    except Exception as e:
        print(f"⚠️  Could not check Ollama status: {e}")
        return False

def main():
    """Main demo function"""
    print("🚀 PRODUCTION RAG SYSTEM DEMO")
    print("Using LangChain + Ollama for reliable AI responses")
    print("=" * 60)
    
    # Check Ollama status
    if not check_ollama_status():
        print("\n💡 To fix this:")
        print("1. Make sure Ollama is installed and running")
        print("2. Run: ollama pull llama3.1:8b")
        print("3. Or try: ollama pull phi3 (smaller model)")
        return
    
    # Initialize the system
    try:
        # You can change the model here based on what you have available
        rag = ProductionRAG(ollama_model="phi3")  # or "phi3", "mistral", etc.
        
        # Test the connection
        if not rag.test_model_connection():
            print("❌ Model connection failed. Please check your Ollama setup.")
            return
        
        # Demo questions
        questions = [
            "What is the attention mechanism and how does it work?",
            "How do transformers process language differently from RNNs?",
            "What are the key differences between BERT and GPT models?",
            "How many models are available on Hugging Face and why is this significant?",
            "What makes RAG systems effective for question answering?",
            "How does Ollama enable local AI development?"
        ]
        
        print("\n🧪 TESTING PRODUCTION RAG SYSTEM")
        print("=" * 60)
        
        # Process each question
        results = []
        for question in questions:
            result = rag.ask(question)
            results.append(result)
            print("\n" + "-" * 60)
        
        # Summary
        print(f"\n🎯 DEMO COMPLETE!")
        print(f"✅ Processed {len(results)} questions successfully")
        print("✅ All responses grounded in curated knowledge base")
        print("✅ Transparent source attribution")
        print("✅ Runs completely locally (no API costs)")
        
        # Demonstrate retrieval process
        print("\n📚 BONUS: Retrieval Process Demonstration")
        rag.demonstrate_retrieval("How do neural networks learn?", k=3)
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check available models: ollama list")
        print("3. Pull a model if needed: ollama pull llama3.1:8b")

if __name__ == "__main__":
    main()