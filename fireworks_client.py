"""
Fireworks.ai Client Module
Handles API interactions for vision and text analysis
"""

import os
import asyncio
import logging
import base64
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Stores analysis results from Fireworks.ai"""
    content: str
    model_used: str
    tokens_used: int
    cost: float
    timestamp: float

class FireworksClient:
    """Client for Fireworks.ai API interactions"""
    
    # Model configurations - Updated with Llama4 Maverick!
    MODELS = {
        "vision": "accounts/fireworks/models/llama4-maverick-instruct-basic",  # Llama4 Maverick for vision!
        "gpt_oss": "accounts/fireworks/models/gpt-oss-120b",
        "qwen3": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        "small": "accounts/fireworks/models/llama-v3p1-8b-instruct"
    }
    
    # Pricing per 1M tokens (update with actual Llama4 pricing)
    PRICING = {
        "vision": {"input": 0.22, "output": 0.88},  # Llama4 Maverick pricing
        "gpt_oss": {"input": 0.15, "output": 0.60},
        "qwen3": {"input": 0.22, "output": 0.88},
        "small": {"input": 0.10, "output": 0.10}
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Fireworks client"""
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        
        if not self.api_key:
            raise ValueError("Fireworks API key not found! Set FIREWORKS_API_KEY environment variable")
        
        # Initialize clients
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )
        
        self.sync_client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )
        
        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}
        
        logger.info("✅ Fireworks client initialized with Llama4 Maverick vision support")
    
    async def analyze_frame(self, 
                           base64_image: str, 
                           prompt: str,
                           max_tokens: int = 500) -> AnalysisResult:
        """Analyze a video frame using Llama4 Maverick vision model
        
        Args:
            base64_image: Base64 encoded image
            prompt: Analysis prompt
            max_tokens: Maximum tokens for response
            
        Returns:
            AnalysisResult with analysis content
        """
        model = self.MODELS["vision"]
        
        try:
            logger.info(f"Analyzing frame with Llama4 Maverick...")
            
            # Construct the message with image - removed unsupported parameters
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=1
                # Removed: top_k, presence_penalty, frequency_penalty (not supported)
            )
            
            # Extract response
            content = response.choices[0].message.content
            
            # Calculate tokens and cost
            input_tokens = 200  # Estimate for vision input
            output_tokens = len(content.split()) * 2
            
            if hasattr(response, 'usage'):
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost = self._calculate_cost("vision", input_tokens, output_tokens)
            
            # Track usage
            self.total_tokens["input"] += input_tokens
            self.total_tokens["output"] += output_tokens
            self.total_cost += cost
            
            logger.info(f"✅ Frame analyzed. Tokens: {input_tokens}+{output_tokens}, Cost: ${cost:.4f}")
            
            return AnalysisResult(
                content=content,
                model_used=model,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame with Llama4 Maverick: {e}")
            raise
    
    async def analyze_text(self,
                          text: str,
                          model_type: str = "gpt_oss",
                          system_prompt: Optional[str] = None,
                          max_tokens: int = 1000) -> AnalysisResult:
        """Analyze text using specified model
        
        Args:
            text: Text to analyze
            model_type: Model to use (gpt_oss, qwen3, or small)
            system_prompt: Optional system prompt
            max_tokens: Maximum response tokens
            
        Returns:
            AnalysisResult with analysis content
        """
        model = self.MODELS.get(model_type, self.MODELS["small"])
        
        try:
            logger.info(f"Analyzing text with {model.split('/')[-1]}...")
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": text})
            
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            # Extract response
            content = response.choices[0].message.content
            
            # Calculate tokens and cost
            input_tokens = len(text.split()) * 2
            output_tokens = len(content.split()) * 2
            
            if hasattr(response, 'usage'):
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost = self._calculate_cost(model_type, input_tokens, output_tokens)
            
            # Track usage
            self.total_tokens["input"] += input_tokens
            self.total_tokens["output"] += output_tokens
            self.total_cost += cost
            
            logger.info(f"✅ Text analyzed. Tokens: {input_tokens}+{output_tokens}, Cost: ${cost:.4f}")
            
            return AnalysisResult(
                content=content,
                model_used=model,
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            raise
    
    def _calculate_cost(self, model_type: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model and tokens"""
        if model_type not in self.PRICING:
            model_type = "small"
        
        pricing = self.PRICING[model_type]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage and cost summary"""
        return {
            "total_input_tokens": self.total_tokens["input"],
            "total_output_tokens": self.total_tokens["output"],
            "total_tokens": self.total_tokens["input"] + self.total_tokens["output"],
            "total_cost_usd": round(self.total_cost, 4)
        }
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = self.sync_client.chat.completions.create(
                model=self.MODELS["small"],
                messages=[{"role": "user", "content": "Say 'Connection successful'"}],
                max_tokens=10
            )
            
            result = response.choices[0].message.content
            logger.info(f"✅ API Test: {result}")
            return True
            
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            return False


# Test with real frame
async def test_llama4_vision():
    """Test Llama4 Maverick vision capabilities"""
    print("\n" + "="*60)
    print("TESTING LLAMA4 MAVERICK VISION")
    print("="*60)
    
    from video_processor import VideoProcessor
    
    # Create a test frame
    processor = VideoProcessor(output_dir="test_frames")
    frames = processor._create_test_frames(1)
    
    if frames:
        frame = frames[0]
        print(f"✅ Using test frame: {frame.frame_path}")
        print(f"   Frame size: {frame.width}x{frame.height}")
        print(f"   Base64 length: {len(frame.base64_image)}")
        
        # Initialize Fireworks client
        client = FireworksClient()
        
        # Test Llama4 Maverick vision
        print("\nTesting Llama4 Maverick vision analysis...")
        
        try:
            result = await client.analyze_frame(
                base64_image=frame.base64_image,
                prompt="""Describe this image in detail. Include:
1. What text do you see?
2. What colors are present?
3. What is the overall composition?
4. Any notable features or patterns?

Be specific and detailed.""",
                max_tokens=500
            )
            
            print(f"\n✅ Llama4 Maverick Vision Analysis Success!")
            print(f"="*40)
            print(f"Response:\n{result.content}")
            print(f"="*40)
            print(f"Model: {result.model_used.split('/')[-1]}")
            print(f"Tokens: {result.tokens_used}")
            print(f"Cost: ${result.cost:.6f}")
            
        except Exception as e:
            print(f"❌ Vision analysis error: {e}")
    
    # Test with a URL image (optional)
    print("\n" + "="*60)
    print("Testing with URL image...")
    
    try:
        # You can also test with URL images
        test_url = "https://images.unsplash.com/photo-1582538885592-e70a5d7ab3d3?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80"
        
        response = await client.async_client.chat.completions.create(
            model=client.MODELS["vision"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": test_url}
                        }
                    ]
                }
            ],
            max_tokens=100,
            temperature=0.6
        )
        
        print(f"✅ URL Image Analysis: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"URL image test skipped: {e}")


# Main test function
async def test_fireworks_client():
    """Test Fireworks client functionality"""
    print("="*60)
    print("FIREWORKS CLIENT TEST WITH LLAMA4 MAVERICK")
    print("="*60)
    
    # Initialize client
    try:
        client = FireworksClient()
        print("✅ Client initialized")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Test connection
    print("\nTesting API connection...")
    if not client.test_connection():
        print("❌ Could not connect to Fireworks API")
        return
    
    # Test text analysis
    print("\nTesting text analysis...")
    test_text = "This is a test video with three people having a conversation about technology."
    
    try:
        result = await client.analyze_text(
            text=f"Analyze this video description: {test_text}",
            model_type="small",
            max_tokens=100
        )
        
        print(f"✅ Text analysis working")
        print(f"  Tokens used: {result.tokens_used}")
        print(f"  Cost: ${result.cost:.6f}")
        
    except Exception as e:
        print(f"❌ Text analysis failed: {e}")
    
    # Test Llama4 vision
    await test_llama4_vision()
    
    # Print final usage summary
    print("\n" + "="*60)
    print("FINAL USAGE SUMMARY")
    print("="*60)
    usage = client.get_usage_summary()
    print(f"Total tokens: {usage['total_tokens']}")
    print(f"Total cost: ${usage['total_cost_usd']}")
    
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    asyncio.run(test_fireworks_client())