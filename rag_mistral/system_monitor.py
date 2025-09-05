import psutil
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Try to import GPU monitoring libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Comprehensive system monitoring for RAG operations"""
    
    def __init__(self):
        self.monitoring = False
        self.stats_history = []
        self.current_operation = None
        self.operation_start_time = None
        self.baseline_stats = self._get_baseline_stats()
        
    def _get_baseline_stats(self) -> Dict:
        """Get baseline system stats before operations"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': self._get_memory_stats(),
            'gpu': self._get_gpu_stats(),
            'torch_memory': self._get_torch_memory_stats()
        }
    
    def _get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent': memory.percent,
            'swap_total_gb': round(swap.total / (1024**3), 2),
            'swap_used_gb': round(swap.used / (1024**3), 2),
            'swap_percent': swap.percent
        }
    
    def _get_gpu_stats(self) -> List[Dict]:
        """Get GPU statistics using multiple methods"""
        gpu_stats = []
        
        # Method 1: GPUtil
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_stats.append({
                        'gpu_id': i,
                        'name': gpu.name,
                        'load_percent': round(gpu.load * 100, 2),
                        'memory_used_mb': round(gpu.memoryUsed, 2),
                        'memory_total_mb': round(gpu.memoryTotal, 2),
                        'memory_percent': round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2),
                        'temperature': gpu.temperature,
                        'method': 'GPUtil'
                    })
            except Exception as e:
                logger.warning(f"GPUtil failed: {e}")
        
        # Method 2: nvidia-ml-py
        if NVML_AVAILABLE and not gpu_stats:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpu_stats.append({
                        'gpu_id': i,
                        'name': name,
                        'load_percent': utilization.gpu,
                        'memory_used_mb': round(memory_info.used / (1024**2), 2),
                        'memory_total_mb': round(memory_info.total / (1024**2), 2),
                        'memory_percent': round((memory_info.used / memory_info.total) * 100, 2),
                        'method': 'nvidia-ml-py'
                    })
            except Exception as e:
                logger.warning(f"nvidia-ml-py failed: {e}")
        
        return gpu_stats
    
    def _get_torch_memory_stats(self) -> Dict:
        """Get PyTorch CUDA memory statistics"""
        if not TORCH_AVAILABLE:
            return {}
        
        stats = {}
        try:
            for i in range(torch.cuda.device_count()):
                device_stats = {
                    'allocated_gb': round(torch.cuda.memory_allocated(i) / (1024**3), 3),
                    'cached_gb': round(torch.cuda.memory_reserved(i) / (1024**3), 3),
                    'max_allocated_gb': round(torch.cuda.max_memory_allocated(i) / (1024**3), 3),
                    'max_cached_gb': round(torch.cuda.max_memory_reserved(i) / (1024**3), 3)
                }
                stats[f'cuda_{i}'] = device_stats
        except Exception as e:
            logger.warning(f"PyTorch memory stats failed: {e}")
        
        return stats
    
    def _get_cpu_stats(self) -> Dict:
        """Get detailed CPU statistics"""
        return {
            'percent_overall': psutil.cpu_percent(interval=0.1),
            'percent_per_core': psutil.cpu_percent(interval=0.1, percpu=True),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'core_count': psutil.cpu_count(),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    def start_operation(self, operation_name: str) -> str:
        """Start monitoring an operation"""
        operation_id = f"{operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.current_operation = {
            'id': operation_id,
            'name': operation_name,
            'start_time': datetime.now(),
            'start_stats': {
                'timestamp': datetime.now().isoformat(),
                'cpu': self._get_cpu_stats(),
                'memory': self._get_memory_stats(),
                'gpu': self._get_gpu_stats(),
                'torch_memory': self._get_torch_memory_stats()
            }
        }
        
        logger.info(f"Started monitoring operation: {operation_name}")
        return operation_id
    
    def end_operation(self, operation_id: str, additional_data: Optional[Dict] = None) -> Dict:
        """End monitoring an operation and return stats"""
        if not self.current_operation or self.current_operation['id'] != operation_id:
            logger.warning(f"No matching operation found for ID: {operation_id}")
            return {}
        
        end_time = datetime.now()
        duration = (end_time - self.current_operation['start_time']).total_seconds()
        
        end_stats = {
            'timestamp': end_time.isoformat(),
            'cpu': self._get_cpu_stats(),
            'memory': self._get_memory_stats(),
            'gpu': self._get_gpu_stats(),
            'torch_memory': self._get_torch_memory_stats()
        }
        
        operation_stats = {
            'operation_id': operation_id,
            'operation_name': self.current_operation['name'],
            'start_time': self.current_operation['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': round(duration, 3),
            'start_stats': self.current_operation['start_stats'],
            'end_stats': end_stats,
            'resource_usage': self._calculate_resource_usage(
                self.current_operation['start_stats'], 
                end_stats
            )
        }
        
        if additional_data:
            operation_stats['additional_data'] = additional_data
        
        self.stats_history.append(operation_stats)
        self.current_operation = None
        
        return operation_stats
    
    def _calculate_resource_usage(self, start_stats: Dict, end_stats: Dict) -> Dict:
        """Calculate resource usage delta between start and end"""
        usage = {}
        
        # Memory usage delta
        start_mem = start_stats.get('memory', {})
        end_mem = end_stats.get('memory', {})
        
        if start_mem and end_mem:
            usage['memory_delta_gb'] = round(
                end_mem.get('used_gb', 0) - start_mem.get('used_gb', 0), 3
            )
            usage['memory_peak_gb'] = end_mem.get('used_gb', 0)
        
        # GPU memory usage delta
        start_gpu = start_stats.get('gpu', [])
        end_gpu = end_stats.get('gpu', [])
        
        if start_gpu and end_gpu and len(start_gpu) == len(end_gpu):
            gpu_deltas = []
            for start_g, end_g in zip(start_gpu, end_gpu):
                delta = {
                    'gpu_id': start_g.get('gpu_id'),
                    'memory_delta_mb': round(
                        end_g.get('memory_used_mb', 0) - start_g.get('memory_used_mb', 0), 2
                    ),
                    'peak_memory_mb': end_g.get('memory_used_mb', 0),
                    'peak_load_percent': end_g.get('load_percent', 0)
                }
                gpu_deltas.append(delta)
            usage['gpu_deltas'] = gpu_deltas
        
        # PyTorch memory usage
        start_torch = start_stats.get('torch_memory', {})
        end_torch = end_stats.get('torch_memory', {})
        
        if start_torch and end_torch:
            torch_deltas = {}
            for device in start_torch:
                if device in end_torch:
                    torch_deltas[device] = {
                        'allocated_delta_gb': round(
                            end_torch[device].get('allocated_gb', 0) - 
                            start_torch[device].get('allocated_gb', 0), 3
                        ),
                        'peak_allocated_gb': end_torch[device].get('max_allocated_gb', 0),
                        'peak_cached_gb': end_torch[device].get('max_cached_gb', 0)
                    }
            usage['torch_deltas'] = torch_deltas
        
        return usage
    
    def get_current_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': self._get_cpu_stats(),
            'memory': self._get_memory_stats(),
            'gpu': self._get_gpu_stats(),
            'torch_memory': self._get_torch_memory_stats()
        }
    
    def get_operation_stats(self, operation_id: str) -> Optional[Dict]:
        """Get stats for a specific operation"""
        for stats in self.stats_history:
            if stats['operation_id'] == operation_id:
                return stats
        return None
    
    def get_all_stats(self) -> List[Dict]:
        """Get all operation statistics"""
        return self.stats_history.copy()
    
    def clear_stats(self):
        """Clear all stored statistics"""
        self.stats_history.clear()
        logger.info("Cleared all operation statistics")
    
    def export_stats(self, filepath: str):
        """Export all statistics to JSON file"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'baseline_stats': self.baseline_stats,
            'operations': self.stats_history,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'gpu_available': GPU_AVAILABLE,
                'torch_available': TORCH_AVAILABLE,
                'nvml_available': NVML_AVAILABLE
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported statistics to {filepath}")

# Global monitor instance
system_monitor = SystemMonitor()
