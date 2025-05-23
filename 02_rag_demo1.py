import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAG:
    def __init__(self):
        print("Loading components...")
        
        # Load embedding model for retrieval
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Use a more reliable model for generation
        print("Loading text generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Set padding token to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Knowledge base - your curated facts
        self.knowledge_base = [
            "Transformers use self-attention mechanisms to process sequences in parallel.",
            "BERT is an encoder-only transformer model designed for understanding tasks.",
            "GPT models are decoder-only transformers designed for text generation.",
            "The attention mechanism allows models to focus on relevant parts of the input.",
            "RAG combines retrieval and generation for better factual accuracy.",
            "Hugging Face hosts over 1.7 million machine learning models.",
            "Fine-tuning adapts pre-trained models to specific tasks or domains.",
            "FAISS is a library for efficient similarity search and clustering."
        ]
        
        # Create embeddings for knowledge base
        print("Creating knowledge base embeddings...")
        self.kb_embeddings = self.embedder.encode(self.knowledge_base)
        
        # Create FAISS index for fast retrieval
        self.index = faiss.IndexFlatIP(self.kb_embeddings.shape[1])
        self.index.add(self.kb_embeddings.astype('float32'))
        
        print("RAG system ready!")
    
    def retrieve(self, query, top_k=2):
        """Retrieve relevant documents from knowledge base"""
        # Convert question to embedding
        query_embedding = self.embedder.encode([query])
        
        # Search for similar documents
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return retrieved documents and their similarity scores
        retrieved_docs = [self.knowledge_base[i] for i in indices[0]]
        return retrieved_docs, scores[0]
    
    def generate_answer(self, query, retrieved_docs):
        """Generate answer using retrieved context with improved stability"""
        # Combine retrieved documents into context
        context = "\n".join(retrieved_docs)
        
        # Create a well-structured prompt
        prompt = f"""Based on the following information, please answer the question.

Information: {context}

Question: {query}

Answer:"""
        
        # Tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response with stable parameters
        try:
            with torch.no_grad():  # Saves memory and prevents gradient calculations
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,  # Generate up to 100 new tokens
                    do_sample=True,      # Enable sampling for variety
                    temperature=0.8,     # Moderate creativity level
                    top_p=0.9,          # Use nucleus sampling for better quality
                    top_k=50,           # Limit vocabulary to top 50 most likely tokens
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Discourage repetition
                    early_stopping=True      # Stop when natural endpoint reached
                )
        except Exception as e:
            # Fallback response if generation fails
            return f"Based on the retrieved information: {context[:200]}... [Generation encountered technical issues, but here's the relevant context]"
        
        # Decode the generated response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer portion
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            # Fallback: return everything after the prompt
            answer = full_response[len(prompt):].strip()
        
        return answer if answer else "I found relevant information but couldn't generate a complete response."
    
    def ask(self, question):
        """Main RAG function that orchestrates retrieval and generation"""
        print(f"\nQuestion: {question}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs, scores = self.retrieve(question)
        print(f"Retrieved context:")
        for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
            print(f"  {i+1}. {doc} (similarity: {score:.3f})")
        
        # Step 2: Generate answer using retrieved context
        answer = self.generate_answer(question, retrieved_docs)
        print(f"Answer: {answer}")
        
        return answer

# Demo section
if __name__ == "__main__":
    # Initialize the RAG system
    rag = SimpleRAG()
    
    # Test with various questions to demonstrate different capabilities
    questions = [
        "What is the attention mechanism?",
        "How do transformers work?",
        "What is the difference between BERT and GPT?",
        "How many models are on Hugging Face?"
    ]
    
    # Process each question and show the complete RAG workflow
    for question in questions:
        rag.ask(question)
        print("-" * 80)