import time
import functools



def timing_decorator(operation_name):
    """
    Decorator to measure execution time and maintain running averages.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
          
            start_time = time.time()
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Update running average
            if not hasattr(self, f'{operation_name}_count'):
                setattr(self, f'{operation_name}_count', 0)
                setattr(self, f'{operation_name}_running_mean', 0.0)
            
            count = getattr(self, f'{operation_name}_count')
            running_mean = getattr(self, f'{operation_name}_running_mean')
            
            # Update running average: new_mean = old_mean + (new_value - old_mean) / (count + 1)
            new_count = count + 1
            new_running_mean = running_mean + (execution_time - running_mean) / new_count
            
            setattr(self, f'{operation_name}_count', new_count)
            setattr(self, f'{operation_name}_running_mean', new_running_mean)
            
            return result
        return wrapper
    return decorator