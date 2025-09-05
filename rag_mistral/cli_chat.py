#!/usr/bin/env python3
# -*- coding: cp1252 -*-
import argparse
import sys
from pathlib import Path
from rag_system import RAGChatbot
from config import *

def print_banner():
    print("="*70)
    print("🚀 High-Performance RAG PDF Chatbot with System Monitoring")
    print("📚 Powered by Mistral-7B + FAISS + A100")
    print("📊 Real-time GPU/CPU/Memory tracking")
    print("="*70)

def format_memory(bytes_val):
    """Format bytes to human readable format"""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f}KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/(1024**2):.1f}MB"
    else:
        return f"{bytes_val/(1024**3):.2f}GB"

def print_detailed_stats(stats):
    """Print comprehensive statistics"""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE SYSTEM STATISTICS")
    print("="*70)
    
    # Performance metrics from last conversation
    if 'stats' in stats and 'performance_metrics' in stats['stats']:
        perf = stats['stats']['performance_metrics']
        print(f"⚡ Last Query Performance:")
        print(f"  • Response Time: {perf.get('response_time_seconds', 0):.3f}s")
        print(f"  • Efficiency Score: {perf.get('efficiency_score', 0)}")
        
        if 'memory_impact' in perf:
            mem = perf['memory_impact']
            print(f"  • Memory Used: {mem.get('memory_used_gb', 0):.3f}GB")
            print(f"  • Peak Memory: {mem.get('peak_memory_gb', 0):.3f}GB")
        
        if 'gpu_impact' in perf and perf['gpu_impact']:
            for i, gpu in enumerate(perf['gpu_impact']):
                print(f"  • GPU {gpu.get('gpu_id', i)} Memory: {gpu.get('memory_used_mb', 0):.1f}MB")
                print(f"  • GPU {gpu.get('gpu_id', i)} Peak Load: {gpu.get('peak_load_percent', 0):.1f}%")
    
    print(f"\n🗂️ Document Statistics:")
    rag_stats = stats.get('rag_system', {})
    print(f"  • Total Documents: {rag_stats.get('total_documents', 0)}")
    print(f"  • PDF Files: {rag_stats.get('pdf_files_indexed', 0)}")
    print(f"  • Conversation Length: {rag_stats.get('conversation_length', 0)}")
    
    # Session statistics
    session = stats.get('current_session', {})
    if session.get('total_conversations', 0) > 0:
        print(f"\n💬 Current Session:")
        print(f"  • Conversations: {session.get('total_conversations', 0)}")
        print(f"  • Duration: {session.get('session_duration_minutes', 0):.1f} minutes")
        print(f"  • Total Tokens: {session.get('total_estimated_tokens', 0):,}")
        
        if 'performance_summary' in session:
            perf_sum = session['performance_summary']
            print(f"  • Avg Response Time: {perf_sum.get('avg_response_time', 0):.3f}s")
            print(f"  • Min Response Time: {perf_sum.get('min_response_time', 0):.3f}s")
            print(f"  • Max Response Time: {perf_sum.get('max_response_time', 0):.3f}s")
        
        if 'memory_summary' in session:
            mem_sum = session['memory_summary']
            print(f"  • Avg Memory/Query: {mem_sum.get('avg_memory_per_query_gb', 0):.3f}GB")
            print(f"  • Peak Memory: {mem_sum.get('peak_memory_usage_gb', 0):.3f}GB")
        
        if 'gpu_summary' in session:
            gpu_sum = session['gpu_summary']
            print(f"  • Avg GPU Memory/Query: {gpu_sum.get('avg_gpu_memory_per_query_mb', 0):.1f}MB")
            print(f"  • Peak GPU Memory: {gpu_sum.get('peak_gpu_memory_mb', 0):.1f}MB")
    
    # Current system resources
    system_resources = stats.get('system_resources', {})
    if system_resources:
        print(f"\n🖥️ Current System State:")
        
        # CPU stats
        if 'cpu' in system_resources:
            cpu = system_resources['cpu']
            print(f"  • CPU Usage: {cpu.get('percent_overall', 0):.1f}%")
            if cpu.get('frequency_mhz'):
                print(f"  • CPU Frequency: {cpu.get('frequency_mhz', 0):.0f}MHz")
        
        # Memory stats
        if 'memory' in system_resources:
            mem = system_resources['memory']
            print(f"  • RAM: {mem.get('used_gb', 0):.1f}GB / {mem.get('total_gb', 0):.1f}GB ({mem.get('percent', 0):.1f}%)")
            if mem.get('swap_total_gb', 0) > 0:
                print(f"  • Swap: {mem.get('swap_used_gb', 0):.1f}GB / {mem.get('swap_total_gb', 0):.1f}GB")
        
        # GPU stats
        if 'gpu' in system_resources and system_resources['gpu']:
            for gpu in system_resources['gpu']:
                print(f"  • {gpu.get('name', 'GPU')}: {gpu.get('load_percent', 0):.1f}% load")
                print(f"    Memory: {gpu.get('memory_used_mb', 0):.0f}MB / {gpu.get('memory_total_mb', 0):.0f}MB")
                if 'temperature' in gpu:
                    print(f"    Temperature: {gpu.get('temperature', 0)}°C")
        
        # PyTorch memory
        if 'torch_memory' in system_resources and system_resources['torch_memory']:
            print(f"  • PyTorch CUDA Memory:")
            for device, mem in system_resources['torch_memory'].items():
                print(f"    {device}: {mem.get('allocated_gb', 0):.2f}GB allocated, {mem.get('cached_gb', 0):.2f}GB cached")

def main():
    parser = argparse.ArgumentParser(description="RAG PDF Chatbot CLI with System Monitoring")
    parser.add_argument("--reindex", action="store_true", help="Force reindex all PDFs")
    parser.add_argument("--stats", action="store_true", help="Show detailed system statistics")
    parser.add_argument("--folder", default=COURSEWORK_FOLDER, help="PDF folder path")
    parser.add_argument("--export", action="store_true", help="Export session data at end")
    args = parser.parse_args()
    
    print_banner()
    
    # Check if coursework folder exists
    if not Path(args.folder).exists():
        print(f"❌ Error: Coursework folder '{args.folder}' not found!")
        print(f"Please create the folder and add your PDF files.")
        sys.exit(1)
    
    # Initialize chatbot
    print("🔧 Initializing RAG system with monitoring...")
    chatbot = RAGChatbot(coursework_folder=args.folder)
    
    # Index documents
    try:
        print("📖 Indexing PDF documents...")
        index_stats = chatbot.index_documents(force_reindex=args.reindex)
        print("✅ PDF indexing completed!")
        if index_stats:
            print(f"   Indexing took {index_stats.get('duration_seconds', 0):.2f}s")
            if 'resource_usage' in index_stats:
                memory_used = index_stats['resource_usage'].get('memory_delta_gb', 0)
                if memory_used > 0:
                    print(f"   Memory used: {memory_used:.2f}GB")
    except Exception as e:
        print(f"❌ Error indexing documents: {e}")
        sys.exit(1)
    
    # Show stats if requested
    if args.stats:
        stats = chatbot.get_detailed_stats()
        print_detailed_stats({'rag_system': stats['rag_metrics'], 
                            'current_session': stats['session_summary'],
                            'system_resources': stats['current_system_state']})
        return
    
    print("\n🎯 RAG Chatbot Ready!")
    print("Commands: 'quit' to exit, 'clear' to reset, 'stats' for detailed stats, 'export' to save session")
    print("-" * 70)
    
    # Chat loop
    try:
        while True:
            try:
                user_input = input("\n💭 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                if user_input.lower() == 'clear':
                    chatbot.clear_conversation()
                    print("🧹 Conversation cleared and new session started!")
                    continue
                
                if user_input.lower() == 'stats':
                    stats = chatbot.get_detailed_stats()
                    print_detailed_stats({'rag_system': stats['rag_metrics'], 
                                        'current_session': stats['session_summary'],
                                        'system_resources': stats['current_system_state']})
                    continue
                
                if user_input.lower() == 'export':
                    export_path = chatbot.export_session_data()
                    print(f"📁 Session data exported to: {export_path}")
                    continue
                
                if not user_input:
                    continue
                
                print("🤔 Processing (monitoring GPU/CPU/Memory)...")
                result = chatbot.chat(user_input)
                
                print(f"\n🤖 Assistant: {result['response']}")
                
                # Display performance metrics
                stats = result.get('stats', {})
                print(f"\n⚡ Performance Metrics:")
                print(f"  • Total Time: {stats.get('response_time', 0):.3f}s")
                print(f"  • Search Time: {stats.get('search_time', 0):.3f}s")
                print(f"  • LLM Time: {stats.get('llm_time', 0):.3f}s")
                print(f"  • Sources Found: {stats.get('num_sources', 0)}")
                
                # Resource usage
                resource_usage = stats.get('resource_usage', {})
                if resource_usage.get('memory_delta_gb', 0) != 0:
                    print(f"  • Memory Used: {resource_usage.get('memory_delta_gb', 0):.3f}GB")
                
                if 'gpu_deltas' in resource_usage:
                    for gpu_delta in resource_usage['gpu_deltas']:
                        if gpu_delta.get('memory_delta_mb', 0) != 0:
                            print(f"  • GPU {gpu_delta.get('gpu_id', 0)} Memory: {gpu_delta.get('memory_delta_mb', 0):.1f}MB")
                
                # Performance score
                perf_metrics = stats.get('performance_metrics', {})
                if perf_metrics.get('efficiency_score', 0) > 0:
                    print(f"  • Efficiency Score: {perf_metrics.get('efficiency_score', 0)}")
                
                # Sources
                if result['sources']:
                    print(f"\n📋 Sources Used:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source.get('source', 'Unknown')} (chunk {source.get('chunk_index', '?')})")
                
                print("-" * 70)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    finally:
        # Export session data if requested
        if args.export:
            try:
                export_path = chatbot.export_session_data()
                print(f"\n📁 Session data exported to: {export_path}")
            except Exception as e:
                print(f"❌ Error exporting session: {e}")
        
        # Show final stats
        try:
            final_stats = chatbot.get_stats()
            session_summary = final_stats.get('current_session', {})
            if session_summary.get('total_conversations', 0) > 0:
                print(f"\n📊 Session Summary:")
                print(f"  • Conversations: {session_summary.get('total_conversations', 0)}")
                print(f"  • Duration: {session_summary.get('session_duration_minutes', 0):.1f} minutes")
                print(f"  • Avg Response Time: {session_summary.get('performance_summary', {}).get('avg_response_time', 0):.3f}s")
        except Exception as e:
            print(f"Warning: Could not display final stats: {e}")
        
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
