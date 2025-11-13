import time
import functools
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TimingStats:
    """
    Dedicated class for managing timing statistics.
    Stores timing data in a structured, organized manner.
    """
    operations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def record(self, operation: str, duration: float):
        """
        Record a timing measurement for an operation.
        
        Args:
            operation: Name of the operation (e.g., "compute_reward", "transition")
            duration: Duration in seconds
        """
        if operation not in self.operations:
            self.operations[operation] = {
                "count": 0,
                "total": 0.0,
                "mean": 0.0
            }
        
        ops = self.operations[operation]
        ops["count"] += 1
        ops["total"] += duration
        
        # Update running mean: new_mean = old_mean + (new_value - old_mean) / (count + 1)
        ops["mean"] = ops["mean"] + (duration - ops["mean"]) / ops["count"]
    
    def get_count(self, operation: str) -> int:
        """Get the number of times an operation was called."""
        return self.operations.get(operation, {}).get("count", 0)
    
    def get_mean(self, operation: str) -> float:
        """Get the mean duration for an operation."""
        return self.operations.get(operation, {}).get("mean", 0.0)
    
    def get_total(self, operation: str) -> float:
        """Get the total time spent on an operation."""
        return self.operations.get(operation, {}).get("total", 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert timing stats to a dictionary format.
        
        Returns:
            Dictionary with operation names as keys and stats as values
        """
        return {
            operation: {
                "count": info["count"],
                "total": info["total"],
                "mean": info["mean"]
            }
            for operation, info in self.operations.items()
        }
    
    def __repr__(self) -> str:
        """String representation of timing stats."""
        if not self.operations:
            return "TimingStats(no operations recorded)"
        
        lines = ["TimingStats:"]
        for operation, info in sorted(self.operations.items()):
            lines.append(f"  {operation}: count={info['count']}, mean={info['mean']:.6f}s, total={info['total']:.6f}s")
        return "\n".join(lines)


def timing_decorator(operation_name):
    """
    Decorator to measure execution time and record in TimingStats object.
    
    Args:
        operation_name: Name of the operation being measured
    
    The class using this decorator must have a 'timing_stats' attribute
    of type TimingStats.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if timing_stats exists, create it if not
            if not hasattr(self, 'timing_stats'):
                self.timing_stats = TimingStats()
            
            # Measure execution time
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record the timing
            self.timing_stats.record(operation_name, execution_time)
            
            return result
        return wrapper
    return decorator