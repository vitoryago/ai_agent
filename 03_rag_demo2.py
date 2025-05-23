import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

class HybridRAG:
    def __init__(self):
        print("🔧 Loading Hybrid RAG System...")
        print("This system combines retrieved knowledge with model knowledge for comprehensive answers")
        
        # Load embedding model for semantic search
        print("📊 Loading embedding model for semantic search...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load generation model - keeping GPT-2 for consistency
        print("🤖 Loading text generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Configure tokenizer properly
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Curated knowledge base - specific domain facts
        # These serve as authoritative, up-to-date information
        self.knowledge_base = [
            "Transformers use self-attention mechanisms to process sequences in parallel, eliminating the need for recurrent connections.",
            "BERT is an encoder-only transformer model designed for understanding tasks like classification and question answering.",
            "GPT models are decoder-only transformers designed for text generation by predicting the next word in a sequence.",
            "The attention mechanism allows models to focus on relevant parts of the input when processing each element in a sequence.",
            "RAG (Retrieval-Augmented Generation) combines retrieval and generation for better factual accuracy by grounding responses in retrieved documents.",
            "Hugging Face hosts over 1.7 million machine learning models, making it the largest repository of open-source AI models.",
            "Fine-tuning adapts pre-trained models to specific tasks or domains by continuing training on task-specific data.",
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
            "Multi-head attention allows transformers to attend to different types of relationships simultaneously.",
            "Positional encoding helps transformers understand the order of words since they process sequences in parallel.",
            "LangGraph is a framework for building multi-agent and stateful applications with language models.",
            "Ollama allows running large language models locally on consumer hardware without internet connectivity."
        ]
        
        # Create embeddings for the knowledge base
        print("📚 Creating knowledge base embeddings...")
        self.kb_embeddings = self.embedder.encode(self.knowledge_base)
        
        # Set up FAISS index for fast similarity search
        print("🔍 Setting up similarity search index...")
        self.index = faiss.IndexFlatIP(self.kb_embeddings.shape[1])
        self.index.add(self.kb_embeddings.astype('float32'))
        
        print(f"✅ Hybrid RAG system ready! Knowledge base contains {len(self.knowledge_base)} curated facts.")
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve the most relevant documents from our curated knowledge base.
        
        This provides specific, authoritative information that will be combined
        with the model's broader knowledge to create comprehensive answers.
        """
        print(f"🔍 Searching for relevant information about: '{query}'")
        
        # Convert query to embedding for similarity search
        query_embedding = self.embedder.encode([query])
        
        # Find most similar facts in our knowledge base
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Extract the actual text and similarity scores
        retrieved_docs = [self.knowledge_base[i] for i in indices[0]]
        similarity_scores = scores[0]
        
        # Show what was retrieved for transparency
        print(f"📖 Retrieved {len(retrieved_docs)} relevant facts:")
        for i, (doc, score) in enumerate(zip(retrieved_docs, similarity_scores)):
            relevance_level = "🔥" if score > 0.6 else "✅" if score > 0.4 else "📝"
            print(f"   {relevance_level} {doc[:70]}... (similarity: {score:.3f})")
        
        return retrieved_docs, similarity_scores
    
    def generate_comprehensive_answer(self, query, retrieved_docs):
        """
        Generate a comprehensive answer that uses both retrieved facts and model knowledge.
        
        This approach allows the model to provide complete, well-rounded answers
        while being grounded in the specific facts we've retrieved.
        """
        
        # Format the retrieved information clearly
        if retrieved_docs:
            context_section = "KEY INFORMATION FROM KNOWLEDGE BASE:\n"
            for i, doc in enumerate(retrieved_docs, 1):
                context_section += f"• {doc}\n"
        else:
            context_section = "No specific information found in knowledge base.\n"
        
        # Create a balanced prompt that encourages comprehensive answers
        # This prompt allows the model to use both retrieved and general knowledge
        prompt = f"""{context_section}

QUESTION: {query}

Please provide a comprehensive answer to this question. Use the key information provided above as your foundation, and enhance it with additional relevant context and explanations to give a complete understanding of the topic. If the key information is limited, supplement it with your general knowledge while clearly indicating what comes from the provided facts versus general knowledge.

COMPREHENSIVE ANSWER:"""
        
        # Tokenize with appropriate length limits
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=400)
        
        print("🤖 Generating comprehensive answer...")
        
        try:
            # Generate with parameters optimized for informative, coherent responses
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,        # Allow longer responses for comprehensive answers
                    do_sample=True,            # Enable sampling for natural language
                    temperature=0.7,           # Balanced creativity and consistency
                    top_p=0.9,                # Good nucleus sampling for quality
                    top_k=50,                 # Reasonable vocabulary restriction
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,    # Slight repetition penalty
                    no_repeat_ngram_size=2     # Prevent immediate repetition
                )
        except Exception as e:
            print(f"⚠️  Generation encountered an issue: {e}")
            # Provide a fallback that combines retrieved facts with a basic explanation
            if retrieved_docs:
                return f"Based on the available information: {' '.join(retrieved_docs[:2])}. This represents the core facts about your question from our knowledge base."
            else:
                return "I don't have specific information about this topic in my knowledge base, and encountered an issue generating a response."
        
        # Extract the generated response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean extraction of just the answer portion
        if "COMPREHENSIVE ANSWER:" in full_response:
            answer = full_response.split("COMPREHENSIVE ANSWER:")[-1].strip()
        else:
            # Fallback extraction
            answer = full_response[len(prompt):].strip()
        
        # Clean up the response
        answer = self._clean_response(answer)
        
        return answer if answer else "I wasn't able to generate a complete response to your question."
    
    def _clean_response(self, response):
        """Clean up generated responses to improve readability and remove artifacts."""
        if not response:
            return ""
        
        # Basic cleaning: normalize whitespace and remove common artifacts
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove incomplete sentences at the end if the response is cut off
        if len(response) > 300:
            sentences = response.split('. ')
            if len(sentences) > 1 and len(sentences[-1]) < 10:  # Last sentence seems incomplete
                response = '. '.join(sentences[:-1]) + '.'
        
        # Remove repeated punctuation
        response = re.sub(r'([.!?])\1+', r'\1', response)
        
        return response
    
    def ask(self, question):
        """
        Main function that demonstrates the hybrid RAG approach.
        
        This combines retrieval of specific facts with the model's general knowledge
        to provide comprehensive, accurate answers.
        """
        print(f"\n" + "="*80)
        print(f"🤔 QUESTION: {question}")
        print("="*80)
        
        # Step 1: Retrieve relevant information from our curated knowledge base
        retrieved_docs, scores = self.retrieve(question)
        
        # Step 2: Generate a comprehensive answer using both retrieved and general knowledge
        answer = self.generate_comprehensive_answer(question, retrieved_docs)
        
        print(f"\n💡 COMPREHENSIVE ANSWER:")
        print(f"{answer}")
        
        # Provide transparency about the information sources
        if retrieved_docs and max(scores) > 0.3:
            print(f"\n📚 This answer incorporates {len(retrieved_docs)} facts from our knowledge base")
            print("🧠 Enhanced with general knowledge for completeness")
        else:
            print(f"\n🧠 This answer is based primarily on general knowledge")
            print("📚 Limited relevant information was found in our specific knowledge base")
        
        return answer
    
    def demonstrate_retrieval_process(self, question):
        """
        Show the retrieval process in detail for educational purposes.
        """
        print(f"\n🔍 RETRIEVAL DEMONSTRATION")
        print(f"Query: '{question}'")
        print("-" * 60)
        
        retrieved_docs, scores = self.retrieve(question, top_k=5)
        
        print(f"\nDetailed retrieval results:")
        for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
            print(f"\n{i+1}. Similarity Score: {score:.4f}")
            print(f"   Content: {doc}")
            
            # Explain what the similarity score means
            if score > 0.7:
                print("   📊 Analysis: Highly relevant - strong semantic similarity")
            elif score > 0.5:
                print("   📊 Analysis: Moderately relevant - good semantic match")
            elif score > 0.3:
                print("   📊 Analysis: Somewhat relevant - partial semantic match")
            else:
                print("   📊 Analysis: Low relevance - weak semantic connection")
    
    def add_domain_knowledge(self, new_facts):
        """
        Dynamically add domain-specific knowledge to the system.
        """
        if isinstance(new_facts, str):
            new_facts = [new_facts]
        
        print(f"📝 Adding {len(new_facts)} new domain-specific facts...")
        
        # Add to knowledge base
        original_size = len(self.knowledge_base)
        self.knowledge_base.extend(new_facts)
        
        # Create embeddings for new facts
        new_embeddings = self.embedder.encode(new_facts)
        
        # Update the search index
        self.index.add(new_embeddings.astype('float32'))
        
        print(f"✅ Knowledge base expanded from {original_size} to {len(self.knowledge_base)} facts")
        
        # Show what was added
        for i, fact in enumerate(new_facts, 1):
            print(f"   + {fact}")

# Demonstration and testing section
if __name__ == "__main__":
    print("🚀 HYBRID RAG SYSTEM - COMBINING RETRIEVAL WITH MODEL KNOWLEDGE")
    print("=" * 80)
    print("This system uses retrieved facts as a foundation and enhances them")
    print("with the model's general knowledge for comprehensive answers.")
    print("=" * 80)
    
    # Initialize the hybrid system
    rag = HybridRAG()
    
    presentation_knowledge = [
        "Vector embeddings represent text as high-dimensional numerical vectors that capture semantic meaning.",
        "Semantic search uses vector similarity to find relevant information based on meaning rather than keyword matching.",
        "Modern RAG systems often use hybrid approaches that combine retrieved context with model knowledge."
    ]
    
    rag.add_domain_knowledge(presentation_knowledge)
    
    print("\n" + "="*80)
    print("🧪 TESTING HYBRID RAG SYSTEM")
    print("="*80)
    
    # Test with questions that benefit from comprehensive answers
    test_questions = [
        "What is the attention mechanism and how does it work?",
        "How do transformers process language?", 
        "What are the key differences between BERT and GPT models?",
        "How many models are available on Hugging Face and why is this significant?",
        "What makes RAG systems effective for question answering?"
    ]
    
    for question in test_questions:
        try:
            rag.ask(question)
            print("\n" + "-"*80)
        except Exception as e:
            print(f"❌ Error processing '{question}': {e}")
            print("-"*80)
    
    print("\n🎯 HYBRID RAG TESTING COMPLETE!")
    print("\nThis system demonstrates:")
    print("✅ Retrieval of relevant domain-specific facts")
    print("✅ Integration with general model knowledge") 
    print("✅ Comprehensive, well-rounded answers")
    print("✅ Transparency about information sources")
    print("✅ Practical applicability for real-world use cases")