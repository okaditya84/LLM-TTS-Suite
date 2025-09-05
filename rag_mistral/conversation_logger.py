import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConversationLogger:
    """Logger for conversations with comprehensive statistics"""
    
    def __init__(self, log_directory: str = "conversation_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Current session data
        self.session_id = self._generate_session_id()
        self.session_start_time = datetime.now()
        self.conversations = []
        self.session_stats = {
            'session_id': self.session_id,
            'start_time': self.session_start_time.isoformat(),
            'total_queries': 0,
            'total_tokens_estimated': 0,
            'total_response_time': 0,
            'operations_stats': []
        }
        
        logger.info(f"ConversationLogger initialized with session ID: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def log_conversation(self, 
                        user_query: str,
                        ai_response: str,
                        operation_stats: Dict,
                        sources: List[Dict],
                        additional_metadata: Optional[Dict] = None) -> str:
        """Log a single conversation exchange with full statistics"""
        
        conversation_id = f"conv_{len(self.conversations) + 1:04d}_{datetime.now().strftime('%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        # Estimate token counts (rough approximation)
        query_tokens = len(user_query.split()) * 1.3  # Rough token estimation
        response_tokens = len(ai_response.split()) * 1.3
        total_tokens = query_tokens + response_tokens
        
        conversation_entry = {
            'conversation_id': conversation_id,
            'session_id': self.session_id,
            'timestamp': timestamp,
            'user_query': user_query,
            'ai_response': ai_response,
            'sources': sources,
            'token_estimates': {
                'query_tokens': round(query_tokens),
                'response_tokens': round(response_tokens),
                'total_tokens': round(total_tokens)
            },
            'operation_stats': operation_stats,
            'performance_metrics': self._extract_performance_metrics(operation_stats),
            'metadata': additional_metadata or {}
        }
        
        self.conversations.append(conversation_entry)
        
        # Update session stats
        self.session_stats['total_queries'] += 1
        self.session_stats['total_tokens_estimated'] += total_tokens
        self.session_stats['total_response_time'] += operation_stats.get('duration_seconds', 0)
        self.session_stats['operations_stats'].append({
            'conversation_id': conversation_id,
            'operation_stats': operation_stats
        })
        
        # Save immediately for data safety
        self._save_session()
        
        logger.info(f"Logged conversation {conversation_id}")
        return conversation_id
    
    def _extract_performance_metrics(self, operation_stats: Dict) -> Dict:
        """Extract key performance metrics from operation stats"""
        metrics = {
            'response_time_seconds': operation_stats.get('duration_seconds', 0),
            'memory_impact': {},
            'gpu_impact': {},
            'efficiency_score': 0
        }
        
        # Memory metrics
        resource_usage = operation_stats.get('resource_usage', {})
        
        if 'memory_delta_gb' in resource_usage:
            metrics['memory_impact'] = {
                'memory_used_gb': resource_usage['memory_delta_gb'],
                'peak_memory_gb': resource_usage.get('memory_peak_gb', 0)
            }
        
        # GPU metrics
        if 'gpu_deltas' in resource_usage:
            gpu_metrics = []
            for gpu_delta in resource_usage['gpu_deltas']:
                gpu_metrics.append({
                    'gpu_id': gpu_delta.get('gpu_id'),
                    'memory_used_mb': gpu_delta.get('memory_delta_mb', 0),
                    'peak_memory_mb': gpu_delta.get('peak_memory_mb', 0),
                    'peak_load_percent': gpu_delta.get('peak_load_percent', 0)
                })
            metrics['gpu_impact'] = gpu_metrics
        
        # PyTorch memory metrics
        if 'torch_deltas' in resource_usage:
            metrics['torch_memory_impact'] = resource_usage['torch_deltas']
        
        # Calculate efficiency score (tokens per second per GB memory used)
        duration = operation_stats.get('duration_seconds', 1)
        memory_used = resource_usage.get('memory_delta_gb', 0.1)
        
        if duration > 0 and memory_used > 0:
            metrics['efficiency_score'] = round(
                (100 / duration) / memory_used, 2  # Arbitrary efficiency metric
            )
        
        return metrics
    
    def _save_session(self):
        """Save current session to JSON file"""
        session_filename = f"{self.session_id}.json"
        session_filepath = self.log_directory / session_filename
        
        # Update session stats
        self.session_stats['end_time'] = datetime.now().isoformat()
        self.session_stats['session_duration_minutes'] = round(
            (datetime.now() - self.session_start_time).total_seconds() / 60, 2
        )
        
        # Calculate averages
        if self.session_stats['total_queries'] > 0:
            self.session_stats['average_response_time'] = round(
                self.session_stats['total_response_time'] / self.session_stats['total_queries'], 3
            )
            self.session_stats['average_tokens_per_query'] = round(
                self.session_stats['total_tokens_estimated'] / self.session_stats['total_queries']
            )
        
        session_data = {
            'session_metadata': self.session_stats,
            'conversations': self.conversations,
            'export_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(session_filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        if not self.conversations:
            return {
                'session_id': self.session_id,
                'total_conversations': 0,
                'status': 'No conversations yet'
            }
        
        # Calculate statistics
        response_times = [conv['operation_stats'].get('duration_seconds', 0) 
                         for conv in self.conversations]
        memory_usage = []
        gpu_usage = []
        
        for conv in self.conversations:
            resource_usage = conv['operation_stats'].get('resource_usage', {})
            if 'memory_delta_gb' in resource_usage:
                memory_usage.append(resource_usage['memory_delta_gb'])
            
            if 'gpu_deltas' in resource_usage:
                for gpu_delta in resource_usage['gpu_deltas']:
                    gpu_usage.append(gpu_delta.get('memory_delta_mb', 0))
        
        summary = {
            'session_id': self.session_id,
            'session_duration_minutes': round(
                (datetime.now() - self.session_start_time).total_seconds() / 60, 2
            ),
            'total_conversations': len(self.conversations),
            'total_estimated_tokens': self.session_stats['total_tokens_estimated'],
            'performance_summary': {
                'avg_response_time': round(sum(response_times) / len(response_times), 3) if response_times else 0,
                'min_response_time': round(min(response_times), 3) if response_times else 0,
                'max_response_time': round(max(response_times), 3) if response_times else 0,
                'total_response_time': round(sum(response_times), 3)
            }
        }
        
        if memory_usage:
            summary['memory_summary'] = {
                'avg_memory_per_query_gb': round(sum(memory_usage) / len(memory_usage), 3),
                'peak_memory_usage_gb': round(max(memory_usage), 3),
                'total_memory_used_gb': round(sum(memory_usage), 3)
            }
        
        if gpu_usage:
            summary['gpu_summary'] = {
                'avg_gpu_memory_per_query_mb': round(sum(gpu_usage) / len(gpu_usage), 2),
                'peak_gpu_memory_mb': round(max(gpu_usage), 2),
                'total_gpu_memory_mb': round(sum(gpu_usage), 2)
            }
        
        return summary
    
    def export_session(self, custom_filename: Optional[str] = None) -> str:
        """Export session with custom filename"""
        if custom_filename:
            export_path = self.log_directory / f"{custom_filename}.json"
        else:
            export_path = self.log_directory / f"{self.session_id}_export.json"
        
        # Add comprehensive summary
        export_data = {
            'export_metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'exporter_version': '1.0',
                'session_summary': self.get_session_summary()
            },
            'session_data': {
                'session_metadata': self.session_stats,
                'conversations': self.conversations
            }
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session exported to {export_path}")
        return str(export_path)
    
    def load_session(self, session_file: str):
        """Load a previous session"""
        session_path = Path(session_file)
        if not session_path.exists():
            session_path = self.log_directory / f"{session_file}.json"
        
        if not session_path.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")
        
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Restore session data
        if 'session_data' in session_data:  # Export format
            self.session_stats = session_data['session_data']['session_metadata']
            self.conversations = session_data['session_data']['conversations']
        else:  # Direct format
            self.session_stats = session_data['session_metadata']
            self.conversations = session_data['conversations']
        
        self.session_id = self.session_stats['session_id']
        logger.info(f"Loaded session {self.session_id} with {len(self.conversations)} conversations")
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict]:
        """Get specific conversation by ID"""
        for conv in self.conversations:
            if conv['conversation_id'] == conversation_id:
                return conv
        return None
    
    def search_conversations(self, query: str, limit: int = 10) -> List[Dict]:
        """Search conversations by query text"""
        results = []
        query_lower = query.lower()
        
        for conv in self.conversations:
            if (query_lower in conv['user_query'].lower() or 
                query_lower in conv['ai_response'].lower()):
                results.append(conv)
                if len(results) >= limit:
                    break
        
        return results
