#!/usr/bin/env python3
"""
Parallel Performance Testing Suite for Optimized RAG System
Tests both single query performance and parallel query handling
"""

import asyncio
import time
import json
import logging
import statistics
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import argparse
import sys

# Try to import optimized components
try:
    from rag_system import OptimizedRAGChatbot
    from config import *
    OPTIMIZED_AVAILABLE = True
except ImportError:
    try:
        from rag_system import RAGChatbot as OptimizedRAGChatbot
        from config import *
        OPTIMIZED_AVAILABLE = False
    except ImportError:
        print("Error: Could not import RAG system. Please check your installation.")
        sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Comprehensive performance testing for RAG system"""
    
    def __init__(self, coursework_folder: str = "coursework"):
        self.coursework_folder = coursework_folder
        self.chatbot = None
        self.test_results = []
        
        # Test queries for benchmarking
        self.test_queries = [
            "What is remote sensing and how does it work?",
            "Explain the applications of machine learning in remote sensing.",
            "What are the different types of satellite sensors?",
            "How is AI being used in remote sensing applications?",
            "What are the challenges in glacier monitoring using remote sensing?",
            "Describe the principles of electromagnetic radiation in remote sensing.",
            "What is the role of GIS in remote sensing data analysis?",
            "How do different spectral bands help in land cover classification?",
            "What are the advantages of hyperspectral imaging?",
            "Explain the process of atmospheric correction in satellite imagery."
        ]
    
    async def initialize_system(self) -> bool:
        """Initialize the RAG system with error handling"""
        try:
            logger.info("Initializing RAG system...")
            start_time = time.time()
            
            self.chatbot = OptimizedRAGChatbot(self.coursework_folder)
            
            # Index documents if needed
            await asyncio.to_thread(self.chatbot.index_documents)
            
            init_time = time.time() - start_time
            logger.info(f"System initialized in {init_time:.3f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def test_single_query_performance(self, num_tests: int = 5) -> Dict[str, Any]:
        """Test single query performance"""
        logger.info(f"Testing single query performance ({num_tests} tests)...")
        
        results = []
        
        for i in range(num_tests):
            query = self.test_queries[i % len(self.test_queries)]
            
            start_time = time.time()
            try:
                if hasattr(self.chatbot, 'chat_async'):
                    result = await self.chatbot.chat_async(query)
                else:
                    result = await asyncio.to_thread(self.chatbot.chat, query)
                
                response_time = time.time() - start_time
                
                test_result = {
                    'test_id': i + 1,
                    'query': query,
                    'response_time': response_time,
                    'success': True,
                    'target_met': response_time < 2.0,
                    'response_length': len(result.get('response', '')),
                    'num_sources': len(result.get('sources', [])),
                    'performance_score': self._calculate_performance_score(response_time)
                }
                
                results.append(test_result)
                
                status = "✓ PASS" if response_time < 2.0 else "✗ SLOW"
                logger.info(f"Test {i+1}: {response_time:.3f}s {status}")
                
            except Exception as e:
                response_time = time.time() - start_time
                results.append({
                    'test_id': i + 1,
                    'query': query,
                    'response_time': response_time,
                    'success': False,
                    'error': str(e),
                    'target_met': False,
                    'performance_score': 0
                })
                logger.error(f"Test {i+1} failed: {e}")
        
        # Calculate statistics
        successful_tests = [r for r in results if r['success']]
        if successful_tests:
            response_times = [r['response_time'] for r in successful_tests]
            
            stats = {
                'total_tests': num_tests,
                'successful_tests': len(successful_tests),
                'success_rate': (len(successful_tests) / num_tests) * 100,
                'avg_response_time': statistics.mean(response_times),
                'median_response_time': statistics.median(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'std_deviation': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'target_met_count': len([r for r in successful_tests if r['target_met']]),
                'target_met_percentage': (len([r for r in successful_tests if r['target_met']]) / len(successful_tests)) * 100,
                'avg_performance_score': statistics.mean([r['performance_score'] for r in successful_tests])
            }
        else:
            stats = {
                'total_tests': num_tests,
                'successful_tests': 0,
                'success_rate': 0,
                'error': 'No successful tests'
            }
        
        return {
            'test_type': 'single_query_performance',
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'individual_results': results
        }
    
    async def test_parallel_query_performance(self, parallel_counts: List[int] = [2, 5, 10, 20]) -> Dict[str, Any]:
        """Test parallel query performance with different concurrency levels"""
        logger.info("Testing parallel query performance...")
        
        parallel_results = []
        
        for count in parallel_counts:
            logger.info(f"Testing with {count} parallel queries...")
            
            try:
                start_time = time.time()
                
                # Prepare queries
                test_queries = (self.test_queries * ((count // len(self.test_queries)) + 1))[:count]
                
                # Execute parallel queries
                if hasattr(self.chatbot, 'parallel_chat_test'):
                    # Use optimized parallel testing
                    result = await self.chatbot.parallel_chat_test(test_queries, num_parallel=count)
                    
                    parallel_result = {
                        'parallel_count': count,
                        'total_time': result['summary']['total_time'],
                        'avg_response_time': result['summary']['avg_response_time'],
                        'max_response_time': result['summary']['max_response_time'],
                        'min_response_time': result['summary']['min_response_time'],
                        'success_rate': result['summary']['success_rate'],
                        'queries_per_second': result['summary']['queries_per_second'],
                        'target_met': result['summary']['target_met'],
                        'individual_results': result['individual_results']
                    }
                else:
                    # Fallback to manual parallel execution
                    if hasattr(self.chatbot, 'chat_async'):
                        tasks = [self.chatbot.chat_async(query) for query in test_queries]
                    else:
                        tasks = [asyncio.to_thread(self.chatbot.chat, query) for query in test_queries]
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    total_time = time.time() - start_time
                    
                    # Process results
                    successful_results = []
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            continue
                        
                        response_time = result.get('stats', {}).get('total_time', total_time / count)
                        successful_results.append({
                            'query': test_queries[i],
                            'response_time': response_time,
                            'success': True
                        })
                    
                    if successful_results:
                        response_times = [r['response_time'] for r in successful_results]
                        parallel_result = {
                            'parallel_count': count,
                            'total_time': total_time,
                            'avg_response_time': statistics.mean(response_times),
                            'max_response_time': max(response_times),
                            'min_response_time': min(response_times),
                            'success_rate': (len(successful_results) / count) * 100,
                            'queries_per_second': count / total_time,
                            'target_met': statistics.mean(response_times) < 2.0,
                            'individual_results': successful_results
                        }
                    else:
                        parallel_result = {
                            'parallel_count': count,
                            'total_time': total_time,
                            'success_rate': 0,
                            'error': 'No successful queries'
                        }
                
                parallel_results.append(parallel_result)
                
                # Log results
                if 'avg_response_time' in parallel_result:
                    status = "✓ PASS" if parallel_result['target_met'] else "✗ SLOW"
                    logger.info(f"Parallel {count}: {parallel_result['avg_response_time']:.3f}s avg, "
                              f"{parallel_result['queries_per_second']:.1f} q/s {status}")
                else:
                    logger.error(f"Parallel {count}: Failed")
                
                # Wait between tests to avoid resource contention
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Parallel test with {count} queries failed: {e}")
                parallel_results.append({
                    'parallel_count': count,
                    'error': str(e),
                    'success_rate': 0
                })
        
        return {
            'test_type': 'parallel_query_performance',
            'timestamp': datetime.now().isoformat(),
            'results': parallel_results
        }
    
    async def test_system_scaling(self) -> Dict[str, Any]:
        """Test how the system scales under load"""
        logger.info("Testing system scaling characteristics...")
        
        scaling_tests = [
            {'name': 'Light Load', 'parallel_queries': 2, 'repetitions': 3},
            {'name': 'Medium Load', 'parallel_queries': 5, 'repetitions': 2},
            {'name': 'Heavy Load', 'parallel_queries': 10, 'repetitions': 2},
            {'name': 'Stress Test', 'parallel_queries': 20, 'repetitions': 1}
        ]
        
        scaling_results = []
        
        for test in scaling_tests:
            logger.info(f"Running {test['name']} test...")
            
            test_results = []
            
            for rep in range(test['repetitions']):
                try:
                    queries = (self.test_queries * ((test['parallel_queries'] // len(self.test_queries)) + 1))[:test['parallel_queries']]
                    
                    start_time = time.time()
                    
                    if hasattr(self.chatbot, 'parallel_chat_test'):
                        result = await self.chatbot.parallel_chat_test(queries, num_parallel=test['parallel_queries'])
                        
                        test_results.append({
                            'repetition': rep + 1,
                            'total_time': result['summary']['total_time'],
                            'avg_response_time': result['summary']['avg_response_time'],
                            'success_rate': result['summary']['success_rate'],
                            'queries_per_second': result['summary']['queries_per_second']
                        })
                    else:
                        # Fallback implementation
                        tasks = [asyncio.to_thread(self.chatbot.chat, query) for query in queries]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        total_time = time.time() - start_time
                        
                        successful = len([r for r in results if not isinstance(r, Exception)])
                        
                        test_results.append({
                            'repetition': rep + 1,
                            'total_time': total_time,
                            'avg_response_time': total_time / test['parallel_queries'],
                            'success_rate': (successful / test['parallel_queries']) * 100,
                            'queries_per_second': test['parallel_queries'] / total_time
                        })
                    
                except Exception as e:
                    logger.error(f"{test['name']} repetition {rep + 1} failed: {e}")
                    test_results.append({
                        'repetition': rep + 1,
                        'error': str(e),
                        'success_rate': 0
                    })
                
                # Brief pause between repetitions
                await asyncio.sleep(0.5)
            
            # Calculate averages for this test
            successful_tests = [r for r in test_results if 'error' not in r]
            if successful_tests:
                scaling_results.append({
                    'test_name': test['name'],
                    'parallel_queries': test['parallel_queries'],
                    'repetitions': test['repetitions'],
                    'avg_total_time': statistics.mean([r['total_time'] for r in successful_tests]),
                    'avg_response_time': statistics.mean([r['avg_response_time'] for r in successful_tests]),
                    'avg_success_rate': statistics.mean([r['success_rate'] for r in successful_tests]),
                    'avg_queries_per_second': statistics.mean([r['queries_per_second'] for r in successful_tests]),
                    'individual_results': test_results
                })
            else:
                scaling_results.append({
                    'test_name': test['name'],
                    'parallel_queries': test['parallel_queries'],
                    'repetitions': test['repetitions'],
                    'error': 'All repetitions failed',
                    'individual_results': test_results
                })
        
        return {
            'test_type': 'system_scaling',
            'timestamp': datetime.now().isoformat(),
            'results': scaling_results
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all performance tests"""
        logger.info("Starting comprehensive performance test suite...")
        
        # Initialize system
        if not await self.initialize_system():
            return {'error': 'Failed to initialize system'}
        
        comprehensive_results = {
            'test_suite': 'comprehensive_performance',
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'optimized_available': OPTIMIZED_AVAILABLE,
                'coursework_folder': self.coursework_folder
            }
        }
        
        try:
            # Test 1: Single query performance
            logger.info("=== Single Query Performance Test ===")
            single_results = await self.test_single_query_performance(num_tests=10)
            comprehensive_results['single_query_test'] = single_results
            
            # Test 2: Parallel query performance
            logger.info("=== Parallel Query Performance Test ===")
            parallel_results = await self.test_parallel_query_performance([2, 5, 10, 15, 20])
            comprehensive_results['parallel_query_test'] = parallel_results
            
            # Test 3: System scaling
            logger.info("=== System Scaling Test ===")
            scaling_results = await self.test_system_scaling()
            comprehensive_results['system_scaling_test'] = scaling_results
            
            # Generate summary
            comprehensive_results['summary'] = self._generate_test_summary(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            comprehensive_results['error'] = str(e)
            return comprehensive_results
    
    def _calculate_performance_score(self, response_time: float) -> float:
        """Calculate performance score (100 = excellent, 0 = poor)"""
        target_time = 2.0
        if response_time <= target_time:
            return 100.0
        elif response_time <= target_time * 2:
            return 100.0 - ((response_time - target_time) / target_time) * 50
        else:
            return max(0.0, 50.0 - ((response_time - target_time * 2) / target_time) * 25)
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            'overall_performance': 'unknown',
            'target_achievement': False,
            'recommendations': []
        }
        
        try:
            # Single query performance
            single_stats = results.get('single_query_test', {}).get('statistics', {})
            if single_stats and 'avg_response_time' in single_stats:
                avg_single_time = single_stats['avg_response_time']
                target_met_pct = single_stats.get('target_met_percentage', 0)
                
                summary['single_query'] = {
                    'avg_response_time': avg_single_time,
                    'target_met_percentage': target_met_pct,
                    'performance_grade': 'A' if avg_single_time < 1.0 else 'B' if avg_single_time < 2.0 else 'C' if avg_single_time < 5.0 else 'F'
                }
                
                if avg_single_time > 2.0:
                    summary['recommendations'].append("Single query performance is below target. Consider model optimization or hardware upgrade.")
            
            # Parallel query performance
            parallel_results = results.get('parallel_query_test', {}).get('results', [])
            if parallel_results:
                best_parallel = min(parallel_results, key=lambda x: x.get('avg_response_time', float('inf')))
                worst_parallel = max(parallel_results, key=lambda x: x.get('avg_response_time', 0))
                
                summary['parallel_query'] = {
                    'best_parallel_time': best_parallel.get('avg_response_time'),
                    'worst_parallel_time': worst_parallel.get('avg_response_time'),
                    'scalability_factor': worst_parallel.get('avg_response_time', 0) / best_parallel.get('avg_response_time', 1),
                    'max_tested_concurrency': max([r.get('parallel_count', 0) for r in parallel_results])
                }
                
                if summary['parallel_query']['scalability_factor'] > 3.0:
                    summary['recommendations'].append("System shows poor scaling under parallel load. Consider async optimization.")
            
            # Overall assessment
            if single_stats and 'avg_response_time' in single_stats:
                if single_stats['avg_response_time'] < 2.0 and single_stats.get('target_met_percentage', 0) > 80:
                    summary['overall_performance'] = 'excellent'
                    summary['target_achievement'] = True
                elif single_stats['avg_response_time'] < 3.0:
                    summary['overall_performance'] = 'good'
                elif single_stats['avg_response_time'] < 5.0:
                    summary['overall_performance'] = 'fair'
                else:
                    summary['overall_performance'] = 'poor'
            
        except Exception as e:
            summary['error'] = f"Failed to generate summary: {e}"
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_test_results_{timestamp}.json"
        
        filepath = Path(filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to {filepath}")
        return str(filepath)

async def main():
    """Main function for running performance tests"""
    parser = argparse.ArgumentParser(description='RAG System Performance Tester')
    parser.add_argument('--coursework', default='coursework', help='Path to coursework folder')
    parser.add_argument('--test-type', choices=['single', 'parallel', 'scaling', 'comprehensive'], 
                       default='comprehensive', help='Type of test to run')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--num-tests', type=int, default=10, help='Number of single query tests')
    parser.add_argument('--parallel-counts', nargs='+', type=int, default=[2, 5, 10, 20], 
                       help='Parallel query counts to test')
    
    args = parser.parse_args()
    
    tester = PerformanceTester(args.coursework)
    
    try:
        if args.test_type == 'single':
            results = await tester.test_single_query_performance(args.num_tests)
        elif args.test_type == 'parallel':
            results = await tester.test_parallel_query_performance(args.parallel_counts)
        elif args.test_type == 'scaling':
            results = await tester.test_system_scaling()
        else:  # comprehensive
            results = await tester.run_comprehensive_test()
        
        # Save results
        output_file = tester.save_results(results, args.output)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PERFORMANCE TEST RESULTS")
        print(f"{'='*60}")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Overall Performance: {summary.get('overall_performance', 'unknown').upper()}")
            print(f"Target Achievement: {'✓ YES' if summary.get('target_achievement', False) else '✗ NO'}")
            
            if 'single_query' in summary:
                sq = summary['single_query']
                print(f"Single Query Avg: {sq.get('avg_response_time', 0):.3f}s (Grade: {sq.get('performance_grade', 'N/A')})")
            
            if 'recommendations' in summary and summary['recommendations']:
                print("\nRecommendations:")
                for rec in summary['recommendations']:
                    print(f"  • {rec}")
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
