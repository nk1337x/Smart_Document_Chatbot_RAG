"""
LLM Handler Module
==================
Manages language model interactions for generating responses.
Supports free models: Groq API (free), Ollama, HuggingFace.
"""

from typing import Optional, List
import os


class LLMHandler:
    """
    A handler for Language Model interactions.
    
    Supports multiple backends:
    - Groq API (FREE - Llama 3, Mixtral - best quality)
    - Ollama (local LLM server)
    - HuggingFace Transformers (Flan-T5)
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based STRICTLY on the provided document context.

Critical Rules:
- Use ONLY the content from the uploaded file to answer questions
- Do NOT use any external knowledge or information outside the provided context
- If the answer is not present in the context, respond EXACTLY with "I don't know."
- Do not include any information from outside the document
- Avoid assumptions, guesses, or added information

Formatting Requirements:
- Present answers in a clear and structured format
- Use bullet points (•) for lists
- Use numbered lists (1., 2., 3.) for step-by-step instructions
- Use short paragraphs for explanations
- Be concise, accurate, and professional"""

    RAG_PROMPT_TEMPLATE = """Context from documents:
{context}

Question: {question}

Instructions:
1. Answer using ONLY the information from the context above
2. If the answer is not in the context, respond with "I don't know."
3. Format your answer with:
   - Bullet points for lists
   - Numbered lists for step-by-step instructions
   - Short, clear paragraphs
4. Be concise and professional"""

    def __init__(self, model_type: str = "groq", model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the LLM handler.
        
        Args:
            model_type: "groq", "ollama", or "huggingface"
            model_name: Specific model name
            api_key: API key for Groq (optional, can use env var)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.model = None
        self.tokenizer = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model based on the specified type."""
        
        if self.model_type == "groq":
            self._init_groq()
        elif self.model_type == "huggingface":
            self._init_huggingface()
        elif self.model_type == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _init_groq(self):
        """Initialize Groq API client (FREE tier available)."""
        try:
            from groq import Groq
            
            # Get API key from parameter or environment
            api_key = self.api_key or os.environ.get("GROQ_API_KEY")
            
            if not api_key:
                raise RuntimeError("Groq API key required. Get free key at: https://console.groq.com/keys")
            
            self.client = Groq(api_key=api_key)
            
            # Use Llama 3 8B by default (fast and good quality)
            if self.model_name is None:
                self.model_name = "llama-3.1-8b-instant"
            
            print(f"Groq initialized with model: {self.model_name}")
            
        except ImportError:
            raise RuntimeError("Groq package not installed. Run: pip install groq")
    
    def _init_huggingface(self):
        """Initialize HuggingFace model (Flan-T5)."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        if self.model_name is None:
            self.model_name = "google/flan-t5-base"
        
        print(f"Loading HuggingFace model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
            
            self.device = device
            print(f"Model loaded on {device}")
            
        except Exception as e:
            print(f"Error: {e}, falling back to smaller model")
            self.model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.device = "cpu"
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            ollama.list()
            
            if self.model_name is None:
                self.model_name = "llama2"
            
            self.ollama_client = ollama
            print(f"Ollama initialized with model: {self.model_name}")
            
        except ImportError:
            raise RuntimeError("Ollama package not installed. Run: pip install ollama")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama. Make sure Ollama is running. Error: {e}")
    
    def generate_response(self, question: str, context: str, 
                         max_length: int = 1024,
                         temperature: float = 0.3) -> str:
        """Generate a response to a question using the provided context."""
        
        prompt = self.RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        
        if self.model_type == "groq":
            return self._generate_groq(prompt, max_length, temperature)
        elif self.model_type == "huggingface":
            return self._generate_huggingface(prompt, max_length, temperature)
        elif self.model_type == "ollama":
            return self._generate_ollama(prompt, max_length, temperature)
        else:
            return "Error: Unknown model type"
    
    def _generate_groq(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate response using Groq API (FREE)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_huggingface(self, prompt: str, max_length: int, 
                              temperature: float) -> str:
        """Generate response using HuggingFace model."""
        import torch
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=20,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_ollama(self, prompt: str, max_length: int,
                         temperature: float) -> str:
        """Generate response using Ollama."""
        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_length
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            return f"Error generating response with Ollama: {str(e)}"
    
    def generate_with_history(self, question: str, context: str,
                             chat_history: List[dict],
                             max_length: int = 512) -> str:
        """
        Generate a response considering chat history.
        
        Args:
            question: Current question
            context: Retrieved context
            chat_history: List of previous messages
            max_length: Maximum response length
            
        Returns:
            Generated response
        """
        # Build conversation context
        history_text = ""
        for msg in chat_history[-4:]:  # Last 4 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        # Enhanced prompt with history
        enhanced_prompt = f"""Previous conversation:
{history_text}

Context from documents:
{context}

Current question: {question}

Please provide a helpful answer based on the context and conversation history."""
        
        if self.model_type == "huggingface":
            return self._generate_huggingface(enhanced_prompt, max_length, 0.7)
        else:
            return self._generate_ollama(enhanced_prompt, max_length, 0.7)
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'device': getattr(self, 'device', 'N/A')
        }
