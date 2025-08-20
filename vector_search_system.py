"""
Vector Search System for Video Analysis
Enables semantic search of video frames using Pinecone and embeddings
"""

import os
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# Vector search dependencies
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not installed. Install with: pip install pinecone")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not installed. Install with: pip install sentence-transformers")

from fireworks_client import FireworksClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FrameEmbedding:
    """Represents a frame with its embedding and metadata"""
    frame_id: str
    timestamp: float
    frame_number: int
    analysis_text: str
    embedding: Optional[List[float]]
    video_path: str
    frame_path: str

@dataclass
class SearchResult:
    """Represents a search result"""
    frame_id: str
    timestamp: float
    frame_number: int
    analysis_text: str
    similarity_score: float
    video_path: str
    frame_path: str

class EmbeddingGenerator:
    """Generates embeddings for text using various methods"""
    
    def __init__(self, method: str = "sentence_transformers"):
        """
        Initialize embedding generator
        
        Args:
            method: "sentence_transformers" or "fireworks" (using text embeddings)
        """
        self.method = method
        self.model = None
        self.fireworks_client = None
        
        if method == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use a model optimized for semantic search
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ SentenceTransformer model loaded: all-MiniLM-L6-v2")
        
        elif method == "fireworks":
            # Use Fireworks for embeddings (if they support embedding models)
            self.fireworks_client = FireworksClient()
            logger.info("✅ Fireworks client initialized for embeddings")
        
        else:
            raise ValueError(f"Embedding method '{method}' not available or dependencies missing")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if self.method == "sentence_transformers" and self.model:
                # Generate embedding using SentenceTransformers
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            
            elif self.method == "fireworks" and self.fireworks_client:
                # For now, we'll use a simple approach since Fireworks might not have embeddings
                # This would need to be replaced with actual embedding API calls
                logger.warning("Fireworks embedding method not fully implemented")
                # Fallback to sentence transformers if available
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
                    embedding = fallback_model.encode(text, convert_to_tensor=False)
                    return embedding.tolist()
                else:
                    raise ValueError("No embedding method available")
            
            else:
                raise ValueError("No valid embedding method configured")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

class PineconeVectorStore:
    """Manages Pinecone vector database operations"""
    
    def __init__(self, api_key: Optional[str] = None, environment: str = "gcp-starter"):
        """
        Initialize Pinecone connection
        
        Args:
            api_key: Pinecone API key (or from env var)
            environment: Pinecone environment
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable.")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = "video-frames"
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        self._setup_index()
        
    def _setup_index(self):
        """Setup or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",  # or "gcp"
                        region="us-east-1"  # adjust based on your preference
                    )
                )
                
                # Wait for index to be ready
                time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats.total_vector_count} vectors")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def upsert_frame_embeddings(self, frame_embeddings: List[FrameEmbedding], batch_size: int = 100):
        """Upload frame embeddings to Pinecone"""
        try:
            vectors = []
            
            for frame_embed in frame_embeddings:
                if frame_embed.embedding is None:
                    logger.warning(f"Skipping frame {frame_embed.frame_id} - no embedding")
                    continue
                
                # Prepare vector data
                vector_data = {
                    "id": frame_embed.frame_id,
                    "values": frame_embed.embedding,
                    "metadata": {
                        "timestamp": frame_embed.timestamp,
                        "frame_number": frame_embed.frame_number,
                        "analysis_text": frame_embed.analysis_text[:1000],  # Truncate for metadata limits
                        "video_path": frame_embed.video_path,
                        "frame_path": frame_embed.frame_path
                    }
                }
                vectors.append(vector_data)
            
            # Upload in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            logger.info(f"✅ Successfully uploaded {len(vectors)} frame embeddings to Pinecone")
            
        except Exception as e:
            logger.error(f"Error uploading to Pinecone: {e}")
            raise
    
    def search_similar_frames(self, query_embedding: List[float], 
                            top_k: int = 10, 
                            filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar frames using vector similarity"""
        try:
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Convert to SearchResult objects
            results = []
            for match in search_results.matches:
                result = SearchResult(
                    frame_id=match.id,
                    timestamp=match.metadata.get("timestamp", 0.0),
                    frame_number=match.metadata.get("frame_number", 0),
                    analysis_text=match.metadata.get("analysis_text", ""),
                    similarity_score=match.score,
                    video_path=match.metadata.get("video_path", ""),
                    frame_path=match.metadata.get("frame_path", "")
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            raise

class VideoVectorSearch:
    """Main class for video frame vector search functionality"""
    
    def __init__(self, 
                 pinecone_api_key: Optional[str] = None,
                 embedding_method: str = "sentence_transformers"):
        """
        Initialize video vector search system
        
        Args:
            pinecone_api_key: Pinecone API key
            embedding_method: Method for generating embeddings
        """
        self.embedding_generator = EmbeddingGenerator(embedding_method)
        self.vector_store = PineconeVectorStore(pinecone_api_key)
        
        logger.info("✅ Video Vector Search System initialized")
    
    def index_video_analysis(self, video_analysis_results: Dict[str, Any]) -> str:
        """
        Index video analysis results for vector search
        
        Args:
            video_analysis_results: Results from VideoAnalysisSystem
            
        Returns:
            Status message
        """
        try:
            logger.info("🔄 Indexing video analysis for vector search...")
            
            frame_embeddings = []
            video_path = video_analysis_results.get("video_path", "unknown")
            
            # Process each frame analysis
            for frame_analysis in video_analysis_results.get("frame_analyses", []):
                analysis_text = frame_analysis.get("analysis", "")
                
                if not analysis_text or "failed" in analysis_text.lower():
                    logger.warning(f"Skipping frame {frame_analysis.get('frame_number')} - no valid analysis")
                    continue
                
                # Generate embedding for analysis text
                try:
                    embedding = self.embedding_generator.generate_embedding(analysis_text)
                    
                    # Create frame embedding object
                    frame_embed = FrameEmbedding(
                        frame_id=f"{video_path}_{frame_analysis.get('frame_number', 0)}_{frame_analysis.get('timestamp', 0)}",
                        timestamp=frame_analysis.get("timestamp", 0.0),
                        frame_number=frame_analysis.get("frame_number", 0),
                        analysis_text=analysis_text,
                        embedding=embedding,
                        video_path=video_path,
                        frame_path=f"frame_{frame_analysis.get('frame_number', 0):06d}.jpg"
                    )
                    
                    frame_embeddings.append(frame_embed)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for frame {frame_analysis.get('frame_number')}: {e}")
                    continue
            
            if not frame_embeddings:
                return "❌ No valid frame embeddings generated"
            
            # Upload to Pinecone
            self.vector_store.upsert_frame_embeddings(frame_embeddings)
            
            logger.info(f"✅ Successfully indexed {len(frame_embeddings)} frames")
            return f"✅ Successfully indexed {len(frame_embeddings)} frames for vector search"
            
        except Exception as e:
            logger.error(f"Error indexing video analysis: {e}")
            return f"❌ Error indexing video: {e}"
    
    def search_frames_by_event(self, 
                              event_description: str,
                              video_path: Optional[str] = None,
                              top_k: int = 5,
                              min_similarity: float = 0.5) -> List[SearchResult]:
        """
        Search for frames containing a specific event
        
        Args:
            event_description: Natural language description of the event to find
            video_path: Optional filter by specific video
            top_k: Number of top results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"🔍 Searching for event: '{event_description}'")
            
            # Generate embedding for the event description
            query_embedding = self.embedding_generator.generate_embedding(event_description)
            
            # Prepare filter if video_path specified
            filter_dict = None
            if video_path:
                filter_dict = {"video_path": {"$eq": video_path}}
            
            # Search for similar frames
            search_results = self.vector_store.search_similar_frames(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more results to filter by similarity
                filter_dict=filter_dict
            )
            
            # Filter by minimum similarity
            filtered_results = [
                result for result in search_results 
                if result.similarity_score >= min_similarity
            ]
            
            # Limit to top_k
            final_results = filtered_results[:top_k]
            
            logger.info(f"✅ Found {len(final_results)} matching frames")
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching for event: {e}")
            return []
    
    def search_timeframes_by_event(self,
                                  event_description: str,
                                  video_path: Optional[str] = None,
                                  window_seconds: float = 10.0,
                                  top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for timeframes (segments) containing a specific event
        
        Args:
            event_description: Natural language description of the event
            video_path: Optional filter by specific video
            window_seconds: Time window around found frames
            top_k: Number of timeframes to return
            
        Returns:
            List of timeframe dictionaries with start/end times
        """
        try:
            # First, search for individual frames
            frame_results = self.search_frames_by_event(
                event_description=event_description,
                video_path=video_path,
                top_k=top_k * 3  # Get more frames to create timeframes
            )
            
            if not frame_results:
                return []
            
            # Group nearby frames into timeframes
            timeframes = []
            
            for result in frame_results[:top_k]:
                start_time = max(0, result.timestamp - window_seconds / 2)
                end_time = result.timestamp + window_seconds / 2
                
                timeframe = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "center_timestamp": result.timestamp,
                    "center_frame": result.frame_number,
                    "description": result.analysis_text,
                    "similarity_score": result.similarity_score,
                    "video_path": result.video_path,
                    "event_query": event_description
                }
                
                timeframes.append(timeframe)
            
            logger.info(f"✅ Found {len(timeframes)} timeframes for event")
            return timeframes
            
        except Exception as e:
            logger.error(f"Error searching timeframes: {e}")
            return []
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed content"""
        try:
            stats = self.vector_store.index.describe_index_stats()
            
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": len(stats.namespaces) if stats.namespaces else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

class VideoSearchInterface:
    """User-friendly interface for video search"""
    
    def __init__(self, pinecone_api_key: Optional[str] = None):
        """Initialize the search interface"""
        self.search_system = VideoVectorSearch(pinecone_api_key)
    
    def add_video_to_search(self, video_analysis_results: Dict[str, Any]) -> str:
        """Add a video's analysis results to the search index"""
        return self.search_system.index_video_analysis(video_analysis_results)
    
    def find_event(self, 
                   event: str, 
                   video: Optional[str] = None,
                   num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find frames containing a specific event
        
        Args:
            event: Description of what to search for (e.g., "person walking", "car driving")
            video: Optional specific video to search in
            num_results: Number of results to return
            
        Returns:
            List of dictionaries with frame information
        """
        results = self.search_system.search_frames_by_event(
            event_description=event,
            video_path=video,
            top_k=num_results
        )
        
        # Convert to user-friendly format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "timestamp": result.timestamp,
                "frame_number": result.frame_number,
                "description": result.analysis_text,
                "confidence": result.similarity_score,
                "video": result.video_path,
                "frame_file": result.frame_path
            })
        
        return formatted_results
    
    def find_timeframes(self,
                       event: str,
                       video: Optional[str] = None,
                       duration: float = 10.0,
                       num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Find time segments containing a specific event
        
        Args:
            event: Description of what to search for
            video: Optional specific video to search in
            duration: Length of each timeframe in seconds
            num_results: Number of timeframes to return
            
        Returns:
            List of timeframe dictionaries
        """
        return self.search_system.search_timeframes_by_event(
            event_description=event,
            video_path=video,
            window_seconds=duration,
            top_k=num_results
        )
    
    def search_stats(self) -> Dict[str, Any]:
        """Get search system statistics"""
        return self.search_system.get_search_statistics()


# Example usage and testing
async def demo_vector_search():
    """Demonstrate the vector search functionality"""
    
    print("="*60)
    print("VIDEO VECTOR SEARCH DEMO")
    print("="*60)
    
    # Check dependencies
    if not PINECONE_AVAILABLE:
        print("❌ Pinecone not available. Install with: pip install pinecone-client")
        return
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ SentenceTransformers not available. Install with: pip install sentence-transformers")
        return
    
    # Initialize search interface
    try:
        search_interface = VideoSearchInterface()
        print("✅ Vector search system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("Make sure PINECONE_API_KEY is set in your environment")
        return
    
    # Example: Load video analysis results (this would come from your video analysis)
    example_analysis = {
        "video_path": "example_video.mp4",
        "frame_analyses": [
            {
                "frame_number": 1,
                "timestamp": 5.0,
                "analysis": "A person walking down a busy street with cars passing by. The scene is well-lit with natural daylight."
            },
            {
                "frame_number": 2,
                "timestamp": 10.0,
                "analysis": "Close-up of a red car stopping at a traffic light. The driver is visible through the windshield."
            },
            {
                "frame_number": 3,
                "timestamp": 15.0,
                "analysis": "People sitting at outdoor café tables, enjoying food and conversation. Trees and umbrellas provide shade."
            }
        ]
    }
    
    # Index the video
    print("\n🔄 Indexing example video...")
    status = search_interface.add_video_to_search(example_analysis)
    print(status)
    
    # Search for events
    print("\n🔍 Searching for events...")
    
    # Search for people walking
    walking_results = search_interface.find_event("person walking on street")
    print(f"\n'Person walking' results: {len(walking_results)}")
    for result in walking_results:
        print(f"  Frame {result['frame_number']} at {result['timestamp']}s (confidence: {result['confidence']:.3f})")
        print(f"    {result['description'][:100]}...")
    
    # Search for cars
    car_results = search_interface.find_event("car vehicle driving")
    print(f"\n'Car' results: {len(car_results)}")
    for result in car_results:
        print(f"  Frame {result['frame_number']} at {result['timestamp']}s (confidence: {result['confidence']:.3f})")
        print(f"    {result['description'][:100]}...")
    
    # Search for timeframes
    timeframes = search_interface.find_timeframes("people eating at restaurant", duration=10.0)
    print(f"\n'People eating' timeframes: {len(timeframes)}")
    for tf in timeframes:
        print(f"  {tf['start_time']:.1f}s - {tf['end_time']:.1f}s (confidence: {tf['similarity_score']:.3f})")
    
    # Get statistics
    stats = search_interface.search_stats()
    print(f"\nSearch index stats: {stats}")
    
    print("\n✅ Vector search demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_vector_search())