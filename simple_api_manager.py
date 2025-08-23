# simple_api_manager.py
import asyncio
import time
import random
from typing import Any, Callable

class SimpleAPIManager:
    """Simple API manager to fix the immediate error"""
    
    def __init__(self):
        self.last_call_time = 0
        self.min_delay = 0.5
        
    async def acquire(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_call_time = time.time()
    
    async def call_with_retry(self, func, **kwargs):
        """Simple retry mechanism"""
        try:
            await self.acquire()
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            if '429' in str(e):
                await asyncio.sleep(2)
                return await func(**kwargs)
            raise e