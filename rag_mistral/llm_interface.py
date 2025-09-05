import logging
from typing import List, Optional, Union
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralLLM:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        self.backend = None
        
        logger.info(f"Initializing Mistral LLM from {model_path}")
        
        # Try different backends in order of preference
        self._initialize_backend()
        
        if self.llm is None:
            raise RuntimeError("Failed to initialize any backend. Please check your installation.")
        
        logger.info(f"Mistral LLM initialized successfully using {self.backend} backend")
    
    def _initialize_backend(self):
        """Try to initialize different backends in order of preference"""
        
        # Option 1: Try vLLM (fastest but can have compatibility issues)
        try:
            self._init_vllm()
            return
        except Exception as e:
            logger.warning(f"vLLM initialization failed: {e}")
        
        # Option 2: Try Transformers with device_map auto (good performance)
        try:
            self._init_transformers_auto()
            return
        except Exception as e:
            logger.warning(f"Transformers auto initialization failed: {e}")
        
        # Option 3: Try Transformers with manual device placement
        try:
            self._init_transformers_manual()
            return
        except Exception as e:
            logger.warning(f"Transformers manual initialization failed: {e}")
        
        # Option 4: Try CPU-only mode
        try:
            self._init_cpu_only()
            return
        except Exception as e:
            logger.error(f"CPU-only initialization failed: {e}")
    
    def _init_vllm(self):
        """Initialize with vLLM"""
        from vllm import LLM, SamplingParams
        from config import VLLM_CONFIG
        
        self.llm = LLM(
            model=self.model_path,
            **VLLM_CONFIG
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            stop=["</s>", "[INST]", "[/INST]"]
        )
        
        self.backend = "vLLM"
    
    def _init_transformers_auto(self):
        """Initialize with Transformers using device_map auto"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.backend = "Transformers-Auto"
        self.llm = self  # Self-reference for compatibility
    
    def _init_transformers_manual(self):
        """Initialize with Transformers using manual device placement"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(device)
        
        self.backend = "Transformers-Manual"
        self.llm = self  # Self-reference for compatibility
    
    def _init_cpu_only(self):
        """Initialize with CPU-only mode"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        self.backend = "CPU-Only"
        self.llm = self  # Self-reference for compatibility
    
    def format_prompt(self, query: str, context: str, conversation_history: Optional[List] = None) -> str:
        """Format prompt for Mistral-7B-Instruct"""
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents. 
        
Instructions:
- Answer based strictly on the provided context
- If the context doesn't contain enough information, say so clearly
- Provide detailed, accurate, and well-structured responses
- Cite specific parts of the context when relevant
- If asked about something not in the context, acknowledge the limitation"""
        
        # Build conversation context if provided
        conversation_context = ""
        if conversation_history:
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        prompt = f"""[INST] {system_prompt}

Context from PDF documents:
{context}

{conversation_context}
Human: {query}

Please provide a comprehensive answer based on the context above. [/INST]"""
        
        return prompt
    
    def generate_response(self, prompt: str, sampling_params=None) -> str:
        """Generate response using the available backend"""
        try:
            if self.backend == "vLLM":
                return self._generate_vllm(prompt, sampling_params)
            else:
                return self._generate_transformers(prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def _generate_vllm(self, prompt: str, sampling_params=None):
        """Generate response using vLLM"""
        if sampling_params is None:
            sampling_params = self.sampling_params
        
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response
    
    def _generate_transformers(self, prompt: str):
        """Generate response using Transformers"""
        import torch
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4000)
        
        # Move to same device as model
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def chat_with_context(self, query: str, context_documents: List, conversation_history: Optional[List] = None) -> str:
        """Generate response with retrieved context"""
        # Combine context from retrieved documents
        context = "\n\n".join([
            f"Document: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
            for doc in context_documents
        ])
        
        # Format prompt
        prompt = self.format_prompt(query, context, conversation_history)
        
        # Generate response
        response = self.generate_response(prompt)
        
        return response
