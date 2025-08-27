"""
Enhanced API Manager with Aggressive Rate Limiting for Fireworks AI
This fixes the 429 rate limit errors in multi-agent discussions
"""

import asyncio
import time
import random
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class ImprovedAPIManager:
    """Enhanced API manager with conservative rate limiting"""
    
    def __init__(self):
        self.last_call_time = 0
        self.min_delay = 5.0  # 5 seconds minimum between calls
        self.consecutive_failures = 0
        self.backoff_multiplier = 1.0
        self.call_count = 0
        
    async def acquire(self):
        """Conservative rate limiting with dynamic backoff"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        # Calculate dynamic delay based on recent failures
        dynamic_delay = self.min_delay * self.backoff_multiplier
        
        # Add extra delay every 10 calls to prevent sustained high rate
        if self.call_count % 10 == 0 and self.call_count > 0:
            dynamic_delay *= 1.5
            logger.info(f"Batch delay: Extended wait after {self.call_count} calls")
        
        if time_since_last < dynamic_delay:
            wait_time = dynamic_delay - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        self.last_call_time = time.time()
        self.call_count += 1
    
    async def call_with_retry(self, func: Callable, max_retries: int = 3, **kwargs) -> Any:
        """Enhanced retry with exponential backoff"""
        
        for attempt in range(max_retries):
            try:
                await self.acquire()
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    result = func(**kwargs)
                
                # Success - reduce backoff gradually
                self.consecutive_failures = max(0, self.consecutive_failures - 1)
                self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.95)
                
                return result
                
            except Exception as e:
                if '429' in str(e):
                    self.consecutive_failures += 1
                    self.backoff_multiplier = min(3.0, 1.2 ** self.consecutive_failures)
                    
                    if attempt < max_retries - 1:
                        # Aggressive exponential backoff for rate limits
                        wait_time = (3 ** attempt) * 5 + random.uniform(2, 8)
                        logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} attempts due to rate limiting")
                        raise e
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    else:
                        raise e
        
        raise Exception(f"Failed after {max_retries} attempts")
