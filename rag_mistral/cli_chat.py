#!/usr/bin/env python3
# -*- coding: cp1252 -*-
"""
Optimized CLI Chat Interface for RAG System
Targets <2s response time with performance monitoring
"""

import asyncio
import sys
import time
import logging
import argparse

# Try to import optimized components first
try:
    from rag_system import OptimizedRAGChatbot
    from config import *
    OPTIMIZED_AVAILABLE = True
    print("🚀 Using optimized RAG system")
except ImportError:
    try:
        from rag_system import RAGChatbot as OptimizedRAGChatbot
        from config import *
        OPTIMIZED_AVAILABLE = False
        print("⚠️  Using standard RAG system (optimizations not available)")
    except ImportError:
        print("❌ Error: Could not import RAG system")
        sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedCLIChat:
    """Optimized CLI interface with performance monitoring"""
    
    def __init__(self, coursework_folder: str = "coursework"):
        self.coursework_folder = coursework_folder
        self.chatbot = None
        self.session_stats = {
            'queries_processed': 0,
            'total_time': 0,
            'responses_under_2s': 0,
            'response_times': []
        }
        
    async def initialize(self) -> bool:
        """Initialize the optimized RAG system"""
        try:
            print("🔧 Initializing optimized RAG system...")
            start_time = time.time()
            
            self.chatbot = OptimizedRAGChatbot(self.coursework_folder)
            
            # Index documents if needed
            print("📚 Indexing documents...")
            await asyncio.to_thread(self.chatbot.index_documents)
            
            init_time = time.time() - start_time
            print(f"✅ System initialized in {init_time:.3f} seconds")
            
            # Display system info
            self._display_system_info()
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            logger.error(f"Initialization error: {e}")
            return False
    
    def _display_system_info(self):
        """Display system configuration and capabilities"""
        print(f"\n{'='*60}")
        print("🎯 OPTIMIZED RAG SYSTEM READY")
        print(f"{'='*60}")
        print(f"Optimizations: {'✅ Enabled' if OPTIMIZED_AVAILABLE else '❌ Disabled'}")
        print(f"Target Response Time: ⏱️  <2 seconds")
        print(f"Coursework Folder: 📁 {self.coursework_folder}")
        print(f"Documents Indexed: 📄 {len(self.chatbot.vector_store.documents)}")
        
        if hasattr(self.chatbot.llm, 'backend'):
            print(f"LLM Backend: 🤖 {self.chatbot.llm.backend}")
        
        if hasattr(self.chatbot.vector_store, 'device'):
            print(f"Vector Store Device: 🔧 {self.chatbot.vector_store.device}")
        
        print(f"{'='*60}\n")
        
        print("💡 Tips:")
        print("  • Type 'help' for commands")
        print("  • Type 'stats' to see performance statistics")
        print("  • Type 'test' to run performance test")
        print("  • Type 'parallel <n>' to test parallel queries")
        print("  • Type 'quit' or 'exit' to end session")
        print()
    
    async def process_query(self, query: str) -> dict:
        """Process a single query with performance tracking"""
        start_time = time.time()
        
        try:
            # Use async chat if available
            if hasattr(self.chatbot, 'chat_async') and OPTIMIZED_AVAILABLE:
                result = await self.chatbot.chat_async(query)
            else:
                result = await asyncio.to_thread(self.chatbot.chat, query)
            
            response_time = time.time() - start_time
            
            # Update session stats
            self._update_session_stats(response_time)
            
            # Add timing info to result
            result['response_time'] = response_time
            result['target_met'] = response_time < 2.0
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Query processing failed: {e}")
            
            return {
                'response': f"Error processing query: {str(e)}",
                'sources': [],
                'response_time': response_time,
                'target_met': False,
                'error': True
            }
    
    def _update_session_stats(self, response_time: float):
        """Update session statistics"""
        self.session_stats['queries_processed'] += 1
        self.session_stats['total_time'] += response_time
        self.session_stats['response_times'].append(response_time)
        
        if response_time < 2.0:
            self.session_stats['responses_under_2s'] += 1
    
    def _display_response(self, result: dict):
        """Display formatted response with performance info"""
        response_time = result.get('response_time', 0)
        target_met = result.get('target_met', False)
        
        # Performance indicator
        if target_met:
            perf_indicator = f"⚡ {response_time:.3f}s"
            perf_color = "32"  # Green
        elif response_time < 5.0:
            perf_indicator = f"⏳ {response_time:.3f}s"
            perf_color = "33"  # Yellow
        else:
            perf_indicator = f"🐌 {response_time:.3f}s"
            perf_color = "31"  # Red
        
        print(f"\n\033[{perf_color}m{perf_indicator}\033[0m")
        print(f"{'─' * 60}")
        
        if result.get('error'):
            print(f"❌ {result['response']}")
        else:
            print(result['response'])
            
            # Show sources if available
            sources = result.get('sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                    source_name = source.get('source', 'Unknown')
                    print(f"  {i}. {source_name}")
                
                if len(sources) > 3:
                    print(f"  ... and {len(sources) - 3} more")
        
        print(f"{'─' * 60}")
    
    def _display_stats(self):
        """Display session performance statistics"""
        stats = self.session_stats
        
        if stats['queries_processed'] == 0:
            print("📊 No queries processed yet")
            return
        
        avg_time = stats['total_time'] / stats['queries_processed']
        target_percentage = (stats['responses_under_2s'] / stats['queries_processed']) * 100
        
        # Calculate performance score
        if avg_time <= 2.0:
            score = 100
            grade = "A+"
        elif avg_time <= 3.0:
            score = 85
            grade = "A"
        elif avg_time <= 5.0:
            score = 70
            grade = "B"
        else:
            score = 50
            grade = "C"
        
        print(f"\n📊 SESSION PERFORMANCE STATISTICS")
        print(f"{'='*50}")
        print(f"Queries Processed: {stats['queries_processed']}")
        print(f"Average Response Time: {avg_time:.3f}s")
        print(f"Target Met (<2s): {stats['responses_under_2s']}/{stats['queries_processed']} ({target_percentage:.1f}%)")
        print(f"Performance Score: {score}/100 (Grade: {grade})")
        
        if stats['response_times']:
            fastest = min(stats['response_times'])
            slowest = max(stats['response_times'])
            print(f"Fastest Response: {fastest:.3f}s")
            print(f"Slowest Response: {slowest:.3f}s")
        
        print(f"{'='*50}")
        
        # Recommendations
        if avg_time > 2.0:
            print("\n💡 Performance Recommendations:")
            if not OPTIMIZED_AVAILABLE:
                print("  • Install optimized components for better performance")
            if avg_time > 5.0:
                print("  • Consider hardware upgrade or model optimization")
            if target_percentage < 50:
                print("  • Review system configuration and resource allocation")
    
    async def run_performance_test(self, num_queries: int = 5):
        """Run a quick performance test"""
        test_queries = [
            "What is remote sensing?",
            "Explain machine learning applications in remote sensing.",
            "What are satellite sensors?",
            "How is AI used in remote sensing?",
            "What is GIS in remote sensing?"
        ]
        
        print(f"🧪 Running performance test with {num_queries} queries...")
        
        results = []
        total_start = time.time()
        
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            print(f"  Test {i+1}/{num_queries}: ", end="", flush=True)
            
            result = await self.process_query(query)
            response_time = result['response_time']
            
            results.append(response_time)
            
            # Quick result indicator
            if response_time < 2.0:
                print(f"✅ {response_time:.3f}s")
            else:
                print(f"❌ {response_time:.3f}s")
        
        total_time = time.time() - total_start
        avg_time = sum(results) / len(results)
        target_met = len([r for r in results if r < 2.0])
        
        print(f"\n🏁 Test Results:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Average Response: {avg_time:.3f}s")
        print(f"  Target Met: {target_met}/{num_queries} ({(target_met/num_queries)*100:.1f}%)")
        print(f"  Performance: {'✅ PASS' if avg_time < 2.0 else '❌ NEEDS IMPROVEMENT'}")
    
    async def run_parallel_test(self, num_parallel: int = 5):
        """Run parallel query test"""
        if not hasattr(self.chatbot, 'parallel_chat_test'):
            print("❌ Parallel testing not available with current system")
            return
        
        test_queries = [
            "What is remote sensing?",
            "Explain AI in remote sensing.",
            "What are satellite sensors?",
            "How does GIS work?",
            "What is machine learning in RS?"
        ]
        
        print(f"🔄 Running parallel test with {num_parallel} concurrent queries...")
        
        try:
            result = await self.chatbot.parallel_chat_test(test_queries, num_parallel=num_parallel)
            
            summary = result.get('summary', {})
            
            print(f"🏁 Parallel Test Results:")
            print(f"  Total Time: {summary.get('total_time', 0):.3f}s")
            print(f"  Average Response: {summary.get('avg_response_time', 0):.3f}s")
            print(f"  Queries/Second: {summary.get('queries_per_second', 0):.1f}")
            print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
            print(f"  Target Met: {'✅ YES' if summary.get('target_met', False) else '❌ NO'}")
            
        except Exception as e:
            print(f"❌ Parallel test failed: {e}")
    def _show_help(self):
        """Display help information"""
        print(f"\n📖 COMMAND HELP")
        print(f"{'='*40}")
        print("Available commands:")
        print("  help          - Show this help message")
        print("  stats         - Show performance statistics")
        print("  test [n]      - Run performance test (default: 5 queries)")
        print("  parallel [n]  - Run parallel test (default: 5 concurrent)")
        print("  clear         - Clear session statistics")
        print("  quit/exit     - End session")
        print()
        print("To ask questions, simply type your query and press Enter.")
        print(f"{'='*40}\n")
    
    async def run_interactive_session(self):
        """Run the main interactive chat session"""
        if not await self.initialize():
            return
        
        print("🎯 Ready for questions! Type 'help' for commands or ask anything about your documents.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self._display_stats()
                    continue
                elif user_input.lower() == 'clear':
                    self.session_stats = {
                        'queries_processed': 0,
                        'total_time': 0,
                        'responses_under_2s': 0,
                        'response_times': []
                    }
                    print("📊 Session statistics cleared")
                    continue
                elif user_input.lower().startswith('test'):
                    parts = user_input.split()
                    num_tests = int(parts[1]) if len(parts) > 1 else 5
                    await self.run_performance_test(num_tests)
                    continue
                elif user_input.lower().startswith('parallel'):
                    parts = user_input.split()
                    num_parallel = int(parts[1]) if len(parts) > 1 else 5
                    await self.run_parallel_test(num_parallel)
                    continue
                
                # Process query
                print("🤔 Processing...", end="", flush=True)
                result = await self.process_query(user_input)
                print("\r" + " " * 20 + "\r", end="")  # Clear "Processing..."
                
                self._display_response(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.error(f"Session error: {e}")
        
        # Show final stats
        if self.session_stats['queries_processed'] > 0:
            print(f"\n📊 Final Session Summary:")
            self._display_stats()
        
        print("\n👋 Thank you for using the Optimized RAG System!")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Optimized RAG CLI Chat Interface')
    parser.add_argument('--coursework', default='coursework', 
                       help='Path to coursework folder (default: coursework)')
    parser.add_argument('--test-only', action='store_true',
                       help='Run performance test only, no interactive session')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test queries for --test-only mode')
    
    args = parser.parse_args()
    
    cli = OptimizedCLIChat(args.coursework)
    
    if args.test_only:
        print("🧪 Running performance test mode...")
        if await cli.initialize():
            await cli.run_performance_test(args.num_tests)
    else:
        await cli.run_interactive_session()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
