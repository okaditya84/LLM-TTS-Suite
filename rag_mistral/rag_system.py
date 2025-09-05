import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime

from config import *
from pdf_processor import PDFProcessor
from vector_store import FAISSVectorStore
from system_monitor import system_monitor
from conversation_logger import ConversationLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import robust LLM interface first, fallback to original
try:
    from llm_interface_robust import MistralLLM
    logger.info("Using robust LLM interface with multiple backend support")
except ImportError:
    try:
        from llm_interface import MistralLLM
        logger.info("Using original LLM interface")
    except ImportError as e:
        logger.error(f"Could not import any LLM interface: {e}")
        raise ImportError("Please install vLLM or use the robust interface")

class RAGChatbot:
    def __init__(self, coursework_folder: str = COURSEWORK_FOLDER):
        self.coursework_folder = coursework_folder
        self.conversation_history = []
        
        # Initialize monitoring and logging
        self.monitor = system_monitor
        self.conversation_logger = ConversationLogger()
        
        logger.info("Initializing RAG Chatbot components...")
        
        # Monitor initialization
        init_op_id = self.monitor.start_operation("rag_initialization")
        
        # Initialize components
        self.pdf_processor = PDFProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        self.vector_store = FAISSVectorStore(
            embedding_model_name=EMBEDDING_MODEL,
            vector_db_path=VECTOR_DB_PATH
        )
        
        self.llm = MistralLLM(model_path=MISTRAL_MODEL_PATH)
        
        # End monitoring initialization
        init_stats = self.monitor.end_operation(init_op_id, {
            'components_initialized': ['pdf_processor', 'vector_store', 'llm'],
            'model_path': MISTRAL_MODEL_PATH,
            'embedding_model': EMBEDDING_MODEL
        })
        
        logger.info("RAG Chatbot initialized successfully!")
        logger.info(f"Initialization took {init_stats.get('duration_seconds', 0):.3f} seconds")
    
    def index_documents(self, force_reindex: bool = False):
        """Index all PDFs in the coursework folder with monitoring"""
        index_op_id = self.monitor.start_operation("document_indexing")
        
        try:
            if force_reindex:
                self.vector_store.clear_index()
            
            # Check if we already have documents
            if len(self.vector_store.documents) > 0 and not force_reindex:
                logger.info(f"Using existing index with {len(self.vector_store.documents)} documents")
                
                index_stats = self.monitor.end_operation(index_op_id, {
                    'action': 'used_existing_index',
                    'document_count': len(self.vector_store.documents),
                    'force_reindex': force_reindex
                })
                return index_stats
            
            logger.info(f"Indexing PDFs from {self.coursework_folder}")
            
            # Process PDFs
            documents = self.pdf_processor.process_pdf_directory(self.coursework_folder)
            
            if not documents:
                raise ValueError(f"No documents found in {self.coursework_folder}")
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Save index
            self.vector_store.save_index()
            
            # End monitoring
            index_stats = self.monitor.end_operation(index_op_id, {
                'action': 'indexed_documents',
                'document_count': len(documents),
                'coursework_folder': self.coursework_folder,
                'force_reindex': force_reindex
            })
            
            logger.info("Document indexing completed")
            logger.info(f"Indexed {len(documents)} documents in {index_stats.get('duration_seconds', 0):.3f} seconds")
            
            return index_stats
            
        except Exception as e:
            self.monitor.end_operation(index_op_id, {
                'action': 'indexing_failed',
                'error': str(e),
                'force_reindex': force_reindex
            })
            raise
    
    def search_documents(self, query: str, k: int = TOP_K_RETRIEVAL) -> tuple:
        """Search for relevant documents with monitoring"""
        search_op_id = self.monitor.start_operation("document_search")
        
        try:
            documents = self.vector_store.similarity_search(query, k=k)
            
            search_stats = self.monitor.end_operation(search_op_id, {
                'query_length': len(query),
                'requested_k': k,
                'returned_count': len(documents),
                'query_preview': query[:100] + "..." if len(query) > 100 else query
            })
            
            return documents, search_stats
            
        except Exception as e:
            self.monitor.end_operation(search_op_id, {
                'error': str(e),
                'query_length': len(query),
                'requested_k': k
            })
            raise
    
    def chat(self, user_query: str) -> Dict:
        """Main chat interface with comprehensive monitoring"""
        total_op_id = self.monitor.start_operation("complete_chat_interaction")
        
        try:
            # Search for relevant context
            relevant_docs, search_stats = self.search_documents(user_query)
            
            if not relevant_docs:
                response = "I couldn't find any relevant information in your PDF documents to answer this question."
                llm_stats = None
            else:
                # Monitor LLM generation
                llm_op_id = self.monitor.start_operation("llm_generation")
                
                try:
                    # Generate response
                    response = self.llm.chat_with_context(
                        query=user_query,
                        context_documents=relevant_docs,
                        conversation_history=self.conversation_history
                    )
                    
                    llm_stats = self.monitor.end_operation(llm_op_id, {
                        'query_length': len(user_query),
                        'response_length': len(response),
                        'context_docs': len(relevant_docs),
                        'model_backend': getattr(self.llm, 'backend', 'unknown')
                    })
                    
                except Exception as e:
                    llm_stats = self.monitor.end_operation(llm_op_id, {
                        'error': str(e),
                        'query_length': len(user_query),
                        'context_docs': len(relevant_docs)
                    })
                    raise
            
            # End total operation monitoring
            total_stats = self.monitor.end_operation(total_op_id, {
                'user_query_length': len(user_query),
                'response_length': len(response),
                'sources_found': len(relevant_docs),
                'search_stats': search_stats,
                'llm_stats': llm_stats
            })
            
            # Prepare sources information
            sources = [doc.metadata for doc in relevant_docs]
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now().isoformat()
            })
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "sources": [doc.metadata.get('source', 'Unknown') for doc in relevant_docs]
            })
            
            # Keep only last 10 messages for memory management
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Log conversation with all stats
            conversation_id = self.conversation_logger.log_conversation(
                user_query=user_query,
                ai_response=response,
                operation_stats=total_stats,
                sources=sources,
                additional_metadata={
                    'search_stats': search_stats,
                    'llm_stats': llm_stats,
                    'model_backend': getattr(self.llm, 'backend', 'unknown')
                }
            )
            
            return {
                "conversation_id": conversation_id,
                "response": response,
                "sources": sources,
                "stats": {
                    "response_time": total_stats.get('duration_seconds', 0),
                    "num_sources": len(relevant_docs),
                    "search_time": search_stats.get('duration_seconds', 0),
                    "llm_time": llm_stats.get('duration_seconds', 0) if llm_stats else 0,
                    "resource_usage": total_stats.get('resource_usage', {}),
                    "performance_metrics": self.conversation_logger.conversations[-1]['performance_metrics']
                }
            }
            
        except Exception as e:
            error_stats = self.monitor.end_operation(total_op_id, {
                'error': str(e),
                'user_query_length': len(user_query),
                'stage': 'chat_interaction'
            })
            
            # Log error conversation
            error_response = f"I encountered an error while processing your request: {str(e)}"
            self.conversation_logger.log_conversation(
                user_query=user_query,
                ai_response=error_response,
                operation_stats=error_stats,
                sources=[],
                additional_metadata={'error': str(e)}
            )
            
            return {
                "conversation_id": self.conversation_logger.conversations[-1]['conversation_id'],
                "response": error_response,
                "sources": [],
                "stats": {
                    "response_time": error_stats.get('duration_seconds', 0),
                    "error": str(e)
                }
            }
    
    def get_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            "rag_system": {
                "total_documents": len(self.vector_store.documents),
                "conversation_length": len(self.conversation_history),
                "pdf_files_indexed": len(set(doc.metadata.get('source', '') for doc in self.vector_store.documents))
            },
            "current_session": self.conversation_logger.get_session_summary(),
            "system_resources": self.monitor.get_current_stats(),
            "all_operations": len(self.monitor.get_all_stats())
        }
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed statistics including all operations"""
        return {
            "session_summary": self.conversation_logger.get_session_summary(),
            "all_operations": self.monitor.get_all_stats(),
            "current_system_state": self.monitor.get_current_stats(),
            "rag_metrics": {
                "total_documents": len(self.vector_store.documents),
                "conversation_count": len(self.conversation_history),
                "indexed_files": len(set(doc.metadata.get('source', '') for doc in self.vector_store.documents))
            }
        }
    
    def export_session_data(self, filename: Optional[str] = None) -> str:
        """Export complete session data including conversations and stats"""
        if filename is None:
            filename = f"rag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Export conversation logs
        conversation_file = self.conversation_logger.export_session(filename)
        
        # Export system monitoring stats
        stats_file = f"conversation_logs/{filename}_system_stats.json"
        self.monitor.export_stats(stats_file)
        
        logger.info(f"Session data exported to {conversation_file} and {stats_file}")
        return conversation_file
    
    def clear_conversation(self):
        """Clear conversation history and start new session"""
        self.conversation_history = []
        self.conversation_logger = ConversationLogger()  # Start new session
        logger.info("Conversation history cleared and new session started")
