import os
import logging
import asyncio
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try optimized imports first, fallback to original
try:
    from config import *
    USE_OPTIMIZED = True
except ImportError:
    from config import *
    USE_OPTIMIZED = False

from pdf_processor import PDFProcessor
from system_monitor import system_monitor
from conversation_logger import ConversationLogger

# Import optimized components
try:
    from vector_store import OptimizedFAISSVectorStore as VectorStore
    from llm_interface import OptimizedMistralLLM as LLMInterface
    logger = logging.getLogger(__name__)
    logger.info("Using optimized RAG components")
except ImportError:
    from vector_store import FAISSVectorStore as VectorStore
    from llm_interface import MistralLLM as LLMInterface
    logger = logging.getLogger(__name__)
    logger.info("Using standard RAG components")

logging.basicConfig(level=logging.INFO)

class OptimizedRAGChatbot:
    """Optimized RAG Chatbot targeting <2s response time with A100 GPU"""
    
    def __init__(self, coursework_folder: str = COURSEWORK_FOLDER):
        self.coursework_folder = coursework_folder
        self.conversation_history = []
        
        # Performance tracking
        self.response_times = []
        self.parallel_test_results = []
        
        # Initialize monitoring and logging
        self.monitor = system_monitor
        self.conversation_logger = ConversationLogger()
        
        logger.info("Initializing Optimized RAG Chatbot...")
        
        # Monitor initialization with optimizations
        init_op_id = self.monitor.start_operation("optimized_rag_initialization")
        
        # Initialize components with optimizations
        self._initialize_components()
        
        # End monitoring initialization
        init_stats = self.monitor.end_operation(init_op_id, {
            'components_initialized': ['pdf_processor', 'vector_store', 'llm'],
            'optimized': USE_OPTIMIZED,
            'model_path': MISTRAL_MODEL_PATH,
            'embedding_model': EMBEDDING_MODEL,
            'performance_features': self._get_enabled_features()
        })
        
        logger.info("Optimized RAG Chatbot initialized successfully!")
        logger.info(f"Initialization took {init_stats.get('duration_seconds', 0):.3f} seconds")
        logger.info(f"Enabled optimizations: {self._get_enabled_features()}")
    
    def _initialize_components(self):
        """Initialize all components with performance optimizations"""
        
        # PDF Processor (standard)
        self.pdf_processor = PDFProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Optimized Vector Store
        device = "cuda" if USE_GPU_EMBEDDINGS and USE_OPTIMIZED else "cpu"
        self.vector_store = VectorStore(
            embedding_model_name=EMBEDDING_MODEL,
            vector_db_path=VECTOR_DB_PATH,
            device=device if hasattr(VectorStore, '__init__') and 'device' in VectorStore.__init__.__code__.co_varnames else None
        )
        
        # Optimized LLM Interface
        max_workers = MAX_PARALLEL_WORKERS if USE_OPTIMIZED else 1
        self.llm = LLMInterface(
            model_path=MISTRAL_MODEL_PATH,
            max_workers=max_workers if hasattr(LLMInterface, '__init__') and 'max_workers' in LLMInterface.__init__.__code__.co_varnames else None
        )
        
        # Additional optimizations
        if hasattr(self.vector_store, 'optimize_for_queries'):
            self.vector_store.optimize_for_queries()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS if USE_OPTIMIZED else 2)
    
    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled optimization features"""
        features = []
        
        if USE_OPTIMIZED:
            features.append("optimized_backends")
        
        if globals().get('ENABLE_RESPONSE_CACHING', False):
            features.append("response_caching")
        
        if globals().get('ENABLE_QUERY_CACHING', False):
            features.append("query_caching")
        
        if globals().get('ASYNC_PROCESSING', False):
            features.append("async_processing")
        
        if globals().get('USE_GPU_EMBEDDINGS', False):
            features.append("gpu_embeddings")
        
        if globals().get('USE_GPU_FAISS', False):
            features.append("gpu_faiss")
        
        return features
    
    def index_documents(self, force_reindex: bool = False):
        """Optimized document indexing"""
        index_op_id = self.monitor.start_operation("optimized_document_indexing")
        
        try:
            if force_reindex:
                self.vector_store.clear_index()
            
            # Check existing documents
            if len(self.vector_store.documents) > 0 and not force_reindex:
                logger.info(f"Using existing optimized index with {len(self.vector_store.documents)} documents")
                
                index_stats = self.monitor.end_operation(index_op_id, {
                    'action': 'used_existing_optimized_index',
                    'document_count': len(self.vector_store.documents),
                    'optimized': True
                })
                return index_stats
            
            logger.info(f"Indexing PDFs from {self.coursework_folder} with optimizations")
            
            # Process PDFs
            documents = self.pdf_processor.process_pdf_directory(self.coursework_folder)
            
            if not documents:
                raise ValueError(f"No documents found in {self.coursework_folder}")
            
            # Add to optimized vector store
            self.vector_store.add_documents(documents)
            
            # Save optimized index
            self.vector_store.save_index()
            
            # End monitoring
            index_stats = self.monitor.end_operation(index_op_id, {
                'action': 'indexed_documents_optimized',
                'document_count': len(documents),
                'optimization_features': self._get_enabled_features()
            })
            
            logger.info(f"Optimized indexing completed in {index_stats.get('duration_seconds', 0):.3f} seconds")
            return index_stats
            
        except Exception as e:
            self.monitor.end_operation(index_op_id, {
                'action': 'indexing_failed',
                'error': str(e)
            })
            raise
    
    async def search_documents_async(self, query: str, k: int = TOP_K_RETRIEVAL) -> tuple:
        """Async optimized document search"""
        search_op_id = self.monitor.start_operation("optimized_document_search")
        
        try:
            start_time = time.time()
            
            # Use optimized search if available
            if hasattr(self.vector_store, 'similarity_search_async'):
                documents = await self.vector_store.similarity_search_async(query, k=k)
            else:
                # Fallback to sync search in executor
                loop = asyncio.get_event_loop()
                documents = await loop.run_in_executor(
                    self.executor,
                    self.vector_store.similarity_search,
                    query,
                    k
                )
            
            search_time = time.time() - start_time
            
            search_stats = self.monitor.end_operation(search_op_id, {
                'query_length': len(query),
                'requested_k': k,
                'returned_count': len(documents),
                'search_time': search_time,
                'async_search': True
            })
            
            return documents, search_stats
            
        except Exception as e:
            self.monitor.end_operation(search_op_id, {
                'error': str(e),
                'async_search': True
            })
            raise
    
    def search_documents(self, query: str, k: int = TOP_K_RETRIEVAL) -> tuple:
        """Sync optimized document search"""
        search_op_id = self.monitor.start_operation("optimized_document_search_sync")
        
        try:
            start_time = time.time()
            
            # Use optimized search
            if hasattr(self.vector_store, 'similarity_search_optimized'):
                documents = self.vector_store.similarity_search_optimized(query, k=k)
            else:
                documents = self.vector_store.similarity_search(query, k=k)
            
            search_time = time.time() - start_time
            
            search_stats = self.monitor.end_operation(search_op_id, {
                'query_length': len(query),
                'requested_k': k,
                'returned_count': len(documents),
                'search_time': search_time,
                'optimized': True
            })
            
            return documents, search_stats
            
        except Exception as e:
            self.monitor.end_operation(search_op_id, {
                'error': str(e)
            })
            raise
    
    async def chat_async(self, user_query: str) -> Dict:
        """Optimized async chat interface targeting <2s response"""
        total_start = time.time()
        total_op_id = self.monitor.start_operation("optimized_chat_async")
        
        try:
            # Async document search
            relevant_docs, search_stats = await self.search_documents_async(user_query)
            
            if not relevant_docs:
                response = "I couldn't find relevant information in your documents."
                llm_stats = None
            else:
                # Async LLM generation
                llm_op_id = self.monitor.start_operation("optimized_llm_generation_async")
                
                try:
                    llm_start = time.time()
                    
                    # Use async generation if available
                    if hasattr(self.llm, 'chat_with_context_async'):
                        response = await self.llm.chat_with_context_async(
                            query=user_query,
                            context_documents=relevant_docs,
                            conversation_history=self.conversation_history
                        )
                    else:
                        # Fallback to sync in executor
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            self.executor,
                            self.llm.chat_with_context,
                            user_query,
                            relevant_docs,
                            self.conversation_history
                        )
                    
                    llm_time = time.time() - llm_start
                    
                    llm_stats = self.monitor.end_operation(llm_op_id, {
                        'query_length': len(user_query),
                        'response_length': len(response),
                        'context_docs': len(relevant_docs),
                        'llm_time': llm_time,
                        'async_generation': True,
                        'backend': getattr(self.llm, 'backend', 'unknown')
                    })
                    
                except Exception as e:
                    llm_stats = self.monitor.end_operation(llm_op_id, {
                        'error': str(e),
                        'async_generation': True
                    })
                    raise
            
            # Calculate total time
            total_time = time.time() - total_start
            self.response_times.append(total_time)
            
            # End total operation monitoring
            total_stats = self.monitor.end_operation(total_op_id, {
                'total_time': total_time,
                'user_query_length': len(user_query),
                'response_length': len(response),
                'sources_found': len(relevant_docs),
                'optimization_target_met': total_time < 2.0,
                'async_processing': True
            })
            
            # Update conversation history
            self._update_conversation_history(user_query, response, relevant_docs)
            
            # Log conversation
            conversation_id = self.conversation_logger.log_conversation(
                user_query=user_query,
                ai_response=response,
                operation_stats=total_stats,
                sources=[doc.metadata for doc in relevant_docs],
                additional_metadata={
                    'total_time': total_time,
                    'optimization_target_met': total_time < 2.0,
                    'async_processing': True
                }
            )
            
            return {
                "conversation_id": conversation_id,
                "response": response,
                "sources": [doc.metadata for doc in relevant_docs],
                "stats": {
                    "total_time": total_time,
                    "search_time": search_stats.get('duration_seconds', 0),
                    "llm_time": llm_stats.get('duration_seconds', 0) if llm_stats else 0,
                    "optimization_target_met": total_time < 2.0,
                    "performance_score": self._calculate_performance_score(total_time),
                    "num_sources": len(relevant_docs)
                }
            }
            
        except Exception as e:
            total_time = time.time() - total_start
            error_stats = self.monitor.end_operation(total_op_id, {
                'error': str(e),
                'total_time': total_time,
                'async_processing': True
            })
            
            return {
                "conversation_id": None,
                "response": f"Error: {str(e)}",
                "sources": [],
                "stats": {
                    "total_time": total_time,
                    "error": str(e)
                }
            }
    
    def chat(self, user_query: str) -> Dict:
        """Sync wrapper for chat functionality"""
        if ASYNC_PROCESSING and USE_OPTIMIZED:
            # Run async version
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.chat_async(user_query))
            finally:
                loop.close()
        else:
            # Use standard sync chat
            return self._chat_sync(user_query)
    
    def _chat_sync(self, user_query: str) -> Dict:
        """Synchronous optimized chat"""
        total_start = time.time()
        total_op_id = self.monitor.start_operation("optimized_chat_sync")
        
        try:
            # Search documents
            relevant_docs, search_stats = self.search_documents(user_query)
            
            if not relevant_docs:
                response = "I couldn't find relevant information."
                llm_stats = None
            else:
                # Generate response
                llm_op_id = self.monitor.start_operation("optimized_llm_generation_sync")
                
                try:
                    llm_start = time.time()
                    response = self.llm.chat_with_context(
                        query=user_query,
                        context_documents=relevant_docs,
                        conversation_history=self.conversation_history
                    )
                    llm_time = time.time() - llm_start
                    
                    llm_stats = self.monitor.end_operation(llm_op_id, {
                        'llm_time': llm_time,
                        'response_length': len(response)
                    })
                    
                except Exception as e:
                    llm_stats = self.monitor.end_operation(llm_op_id, {'error': str(e)})
                    raise
            
            total_time = time.time() - total_start
            self.response_times.append(total_time)
            
            # End monitoring
            total_stats = self.monitor.end_operation(total_op_id, {
                'total_time': total_time,
                'optimization_target_met': total_time < 2.0
            })
            
            # Update conversation
            self._update_conversation_history(user_query, response, relevant_docs)
            
            return {
                "response": response,
                "sources": [doc.metadata for doc in relevant_docs],
                "stats": {
                    "total_time": total_time,
                    "optimization_target_met": total_time < 2.0,
                    "performance_score": self._calculate_performance_score(total_time)
                }
            }
            
        except Exception as e:
            total_time = time.time() - total_start
            self.monitor.end_operation(total_op_id, {'error': str(e), 'total_time': total_time})
            return {
                "response": f"Error: {str(e)}",
                "sources": [],
                "stats": {"total_time": total_time, "error": str(e)}
            }
    
    async def parallel_chat_test(self, queries: List[str], num_parallel: int = 10) -> Dict:
        """Test parallel query processing performance"""
        logger.info(f"Starting parallel test with {num_parallel} queries")
        
        test_start = time.time()
        test_op_id = self.monitor.start_operation("parallel_chat_test")
        
        try:
            # Prepare test queries (repeat if needed)
            test_queries = (queries * ((num_parallel // len(queries)) + 1))[:num_parallel]
            
            # Execute parallel queries
            if hasattr(self.llm, 'chat_batch_async') and hasattr(self.vector_store, 'batch_similarity_search_async'):
                results = await self._parallel_test_optimized(test_queries)
            else:
                results = await self._parallel_test_standard(test_queries)
            
            total_test_time = time.time() - test_start
            
            # Calculate statistics
            response_times = [r['stats']['total_time'] for r in results if 'stats' in r]
            avg_time = sum(response_times) / len(response_times) if response_times else 0
            max_time = max(response_times) if response_times else 0
            min_time = min(response_times) if response_times else 0
            
            # Success rate
            successful_queries = len([r for r in results if 'error' not in r.get('stats', {})])
            success_rate = (successful_queries / num_parallel) * 100
            
            test_stats = self.monitor.end_operation(test_op_id, {
                'num_parallel_queries': num_parallel,
                'total_test_time': total_test_time,
                'avg_response_time': avg_time,
                'max_response_time': max_time,
                'min_response_time': min_time,
                'success_rate': success_rate,
                'queries_per_second': num_parallel / total_test_time,
                'optimization_target_met': avg_time < 2.0
            })
            
            # Store results
            self.parallel_test_results.append(test_stats)
            
            logger.info(f"Parallel test completed: {avg_time:.3f}s avg, {success_rate:.1f}% success")
            
            return {
                'test_stats': test_stats,
                'individual_results': results,
                'summary': {
                    'total_time': total_test_time,
                    'avg_response_time': avg_time,
                    'max_response_time': max_time,
                    'min_response_time': min_time,
                    'success_rate': success_rate,
                    'queries_per_second': num_parallel / total_test_time,
                    'target_met': avg_time < 2.0
                }
            }
            
        except Exception as e:
            self.monitor.end_operation(test_op_id, {'error': str(e)})
            raise
    
    async def _parallel_test_optimized(self, queries: List[str]) -> List[Dict]:
        """Optimized parallel test using batch processing"""
        
        # Batch document search
        search_start = time.time()
        docs_lists = await self.vector_store.batch_similarity_search_async(queries, k=TOP_K_RETRIEVAL)
        search_time = time.time() - search_start
        
        # Batch LLM generation
        llm_start = time.time()
        responses = await self.llm.chat_batch_async(
            queries=queries,
            context_documents_list=docs_lists,
            conversation_histories=[self.conversation_history] * len(queries)
        )
        llm_time = time.time() - llm_start
        
        # Prepare results
        results = []
        for i, (query, response, docs) in enumerate(zip(queries, responses, docs_lists)):
            total_time = (search_time + llm_time) / len(queries)  # Approximate per-query time
            
            results.append({
                'query': query,
                'response': response,
                'sources': [doc.metadata for doc in docs],
                'stats': {
                    'total_time': total_time,
                    'search_time': search_time / len(queries),
                    'llm_time': llm_time / len(queries),
                    'batch_optimized': True
                }
            })
        
        return results
    
    async def _parallel_test_standard(self, queries: List[str]) -> List[Dict]:
        """Standard parallel test using asyncio gather"""
        
        tasks = [self.chat_async(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'query': queries[i],
                    'response': f"Error: {str(result)}",
                    'sources': [],
                    'stats': {'error': str(result), 'total_time': 0}
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _update_conversation_history(self, query: str, response: str, docs: List):
        """Update conversation history efficiently"""
        timestamp = datetime.now().isoformat()
        
        self.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": timestamp
        })
        
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in docs]
        })
        
        # Keep only recent messages for performance
        if len(self.conversation_history) > 8:  # 4 conversation turns
            self.conversation_history = self.conversation_history[-8:]
    
    def _calculate_performance_score(self, response_time: float) -> float:
        """Calculate performance score (100 = target met, 0 = very slow)"""
        target_time = 2.0  # 2 second target
        if response_time <= target_time:
            return 100.0
        elif response_time <= target_time * 2:
            return 100.0 - ((response_time - target_time) / target_time) * 50
        else:
            return max(0.0, 50.0 - ((response_time - target_time * 2) / target_time) * 25)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.response_times:
            return {"error": "No performance data available"}
        
        avg_time = sum(self.response_times) / len(self.response_times)
        target_met_count = len([t for t in self.response_times if t < 2.0])
        target_met_percentage = (target_met_count / len(self.response_times)) * 100
        
        summary = {
            "performance_metrics": {
                "total_queries": len(self.response_times),
                "avg_response_time": avg_time,
                "max_response_time": max(self.response_times),
                "min_response_time": min(self.response_times),
                "target_met_percentage": target_met_percentage,
                "performance_score": self._calculate_performance_score(avg_time)
            },
            "optimization_status": {
                "optimized_components": USE_OPTIMIZED,
                "enabled_features": self._get_enabled_features(),
                "target_achieved": avg_time < 2.0
            },
            "system_stats": self.llm.get_performance_stats() if hasattr(self.llm, 'get_performance_stats') else {},
            "vector_stats": self.vector_store.get_performance_stats() if hasattr(self.vector_store, 'get_performance_stats') else {}
        }
        
        if self.parallel_test_results:
            summary["parallel_test_results"] = self.parallel_test_results
        
        return summary
    
    def export_performance_report(self, filename: Optional[str] = None) -> str:
        """Export detailed performance report"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_summary": self.get_performance_summary(),
            "raw_response_times": self.response_times,
            "parallel_test_results": self.parallel_test_results,
            "system_configuration": {
                "model_path": MISTRAL_MODEL_PATH,
                "embedding_model": EMBEDDING_MODEL,
                "optimized": USE_OPTIMIZED,
                "enabled_features": self._get_enabled_features()
            }
        }
        
        filepath = Path(CACHE_DIR) / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report exported to {filepath}")
        return str(filepath)
    
    def clear_performance_data(self):
        """Clear performance tracking data"""
        self.response_times.clear()
        self.parallel_test_results.clear()
        logger.info("Performance data cleared")
