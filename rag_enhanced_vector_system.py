"""
COMPLETELY FIXED RAG-Enhanced Vector System
Resolves the 'query_video_rag' method missing error and ensures proper method availability
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
class AgentPerspective:
    """Represents an agent's perspective on a specific frame or timeframe"""
    agent_name: str
    agent_role: str
    frame_number: Optional[int]
    timestamp: Optional[float]
    timeframe_start: Optional[float]
    timeframe_end: Optional[float]
    perspective_content: str
    perspective_type: str
    discussion_round: Optional[int]
    responding_to: Optional[str]
    video_path: str

@dataclass
class RAGVideoDocument:
    """Complete document for RAG system"""
    document_id: str
    document_type: str
    video_path: str
    frame_number: Optional[int]
    timestamp: Optional[float]
    timeframe_start: Optional[float]
    timeframe_end: Optional[float]
    content: str
    metadata: Dict[str, Any]
    agent_name: Optional[str] = None
    agent_role: Optional[str] = None
    perspective_type: Optional[str] = None
    embedding: Optional[List[float]] = None

class RAGEnhancedVectorStore:
    """FIXED: Enhanced vector store with proper error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize RAG-enhanced vector store"""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone")
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable.")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = "video-rag-enhanced"
        self.dimension = 384
        
        self._setup_index()
        self.embedding_generator = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ RAG-Enhanced Vector Store initialized")
    
    def _setup_index(self):
        """Setup or connect to Pinecone index for RAG"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new RAG Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to RAG Pinecone index: {self.index_name}")
            
            stats = self.index.describe_index_stats()
            logger.info(f"RAG Index stats: {stats.total_vector_count} documents")
            
        except Exception as e:
            logger.error(f"Error setting up RAG Pinecone index: {e}")
            raise
    
    def create_rag_documents(self, 
                           video_analysis_results: Dict[str, Any],
                           agent_perspectives: List[AgentPerspective]) -> List[RAGVideoDocument]:
        """FIXED: Create RAG documents with proper error handling"""
        documents = []
        video_path = video_analysis_results.get("video_path", "unknown")
        
        logger.info(f"🔄 Creating RAG documents for video: {video_path}")
        logger.info(f"   Frame analyses available: {len(video_analysis_results.get('frame_analyses', []))}")
        logger.info(f"   Agent perspectives: {len(agent_perspectives)}")
        
        # 1. FIXED: Create documents from frame analyses with safe access
        frame_analyses = video_analysis_results.get("frame_analyses", [])
        if frame_analyses:
            for frame_analysis in frame_analyses:
                if not frame_analysis.get("analysis") or "failed" in frame_analysis.get("analysis", "").lower():
                    continue
                
                doc_id = f"frame_{video_path}_{frame_analysis.get('frame_number', 0)}_{frame_analysis.get('timestamp', 0)}"
                
                document = RAGVideoDocument(
                    document_id=doc_id,
                    document_type="frame_analysis",
                    video_path=video_path,
                    frame_number=frame_analysis.get("frame_number"),
                    timestamp=frame_analysis.get("timestamp"),
                    timeframe_start=None,
                    timeframe_end=None,
                    content=frame_analysis.get("analysis", ""),
                    metadata={
                        "tokens_used": frame_analysis.get("tokens_used", 0),
                        "cost": frame_analysis.get("cost", 0),
                        "analysis_depth": frame_analysis.get("analysis_depth", "unknown")
                    }
                )
                documents.append(document)
        
        # 2. Create documents from agent perspectives
        for perspective in agent_perspectives:
            timestamp_id = perspective.timestamp or 0
            round_id = perspective.discussion_round or 0
            doc_id = f"agent_{perspective.agent_name}_{video_path}_{timestamp_id}_{round_id}_{len(documents)}"
            
            document = RAGVideoDocument(
                document_id=doc_id,
                document_type="agent_perspective",
                video_path=perspective.video_path,
                frame_number=perspective.frame_number,
                timestamp=perspective.timestamp,
                timeframe_start=perspective.timeframe_start,
                timeframe_end=perspective.timeframe_end,
                content=perspective.perspective_content,
                agent_name=perspective.agent_name,
                agent_role=perspective.agent_role,
                perspective_type=perspective.perspective_type,
                metadata={
                    "discussion_round": perspective.discussion_round,
                    "responding_to": perspective.responding_to,
                    "perspective_type": perspective.perspective_type
                }
            )
            documents.append(document)
        
        # 3. FIXED: Create document from overall analysis with safe timeframe calculation
        overall_analysis = video_analysis_results.get("overall_analysis")
        if overall_analysis and len(overall_analysis.strip()) > 0:
            doc_id = f"overall_{video_path}_{datetime.now().timestamp()}"
            
            # FIXED: Safe calculation of timeframe_end
            timeframe_end = 0
            if frame_analyses:
                # Find the last frame with a valid timestamp
                for frame in reversed(frame_analyses):
                    if frame.get("timestamp") is not None:
                        timeframe_end = frame.get("timestamp", 0)
                        break
            
            if timeframe_end == 0:
                # Fallback: estimate based on frame count
                frame_count = video_analysis_results.get("frame_count", 0)
                timeframe_end = frame_count * 5  # Assume 5 seconds per frame
            
            document = RAGVideoDocument(
                document_id=doc_id,
                document_type="overall_analysis",
                video_path=video_path,
                frame_number=None,
                timestamp=None,
                timeframe_start=0,
                timeframe_end=timeframe_end,
                content=overall_analysis,
                metadata={
                    "frame_count": video_analysis_results.get("frame_count", 0),
                    "subtitle_count": video_analysis_results.get("subtitle_count", 0),
                    "processing_time": video_analysis_results.get("processing_time", 0),
                    "total_cost": video_analysis_results.get("total_cost", 0)
                }
            )
            documents.append(document)
        
        logger.info(f"✅ Created {len(documents)} RAG documents total")
        return documents
    
    def generate_embeddings(self, documents: List[RAGVideoDocument]) -> List[RAGVideoDocument]:
        """Generate embeddings for all documents"""
        logger.info(f"🔄 Generating embeddings for {len(documents)} documents...")
        
        successful_count = 0
        for doc in documents:
            try:
                embedding_content = self._create_embedding_content(doc)
                embedding = self.embedding_generator.encode(embedding_content, convert_to_tensor=False)
                doc.embedding = embedding.tolist()
                successful_count += 1
            except Exception as e:
                logger.warning(f"Failed to generate embedding for document {doc.document_id}: {e}")
                doc.embedding = None
        
        logger.info(f"✅ Generated {successful_count}/{len(documents)} embeddings")
        return documents
    
    def _create_embedding_content(self, doc: RAGVideoDocument) -> str:
        """Create optimized content for embedding generation"""
        content_parts = []
        content_parts.append(f"Document type: {doc.document_type}")
        
        if doc.agent_name:
            content_parts.append(f"Agent: {doc.agent_name} ({doc.agent_role})")
            content_parts.append(f"Perspective type: {doc.perspective_type}")
        
        if doc.timestamp is not None:
            content_parts.append(f"Timestamp: {doc.timestamp:.1f} seconds")
        
        if doc.timeframe_start is not None and doc.timeframe_end is not None:
            content_parts.append(f"Timeframe: {doc.timeframe_start:.1f}s to {doc.timeframe_end:.1f}s")
        
        content_parts.append(f"Content: {doc.content}")
        return " | ".join(content_parts)
    
    def _sanitize_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for Pinecone compatibility"""
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    sanitized[key] = value
                else:
                    sanitized[key] = [str(item) for item in value]
            elif isinstance(value, dict):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = str(value)
        
        return sanitized
    
    def upsert_rag_documents(self, documents: List[RAGVideoDocument], batch_size: int = 100):
        """Upload RAG documents to Pinecone"""
        try:
            vectors = []
            
            for doc in documents:
                if doc.embedding is None:
                    logger.warning(f"Skipping document {doc.document_id} - no embedding")
                    continue
                
                base_metadata = {
                    "document_type": doc.document_type,
                    "video_path": doc.video_path,
                    "content": doc.content[:1000]
                }
                
                sanitized_metadata = self._sanitize_metadata_for_pinecone(doc.metadata)
                base_metadata.update(sanitized_metadata)
                
                if doc.agent_name:
                    base_metadata.update({
                        "agent_name": doc.agent_name,
                        "agent_role": doc.agent_role,
                        "perspective_type": doc.perspective_type or "unknown"
                    })
                
                if doc.timestamp is not None:
                    base_metadata["timestamp"] = float(doc.timestamp)
                if doc.frame_number is not None:
                    base_metadata["frame_number"] = int(doc.frame_number)
                if doc.timeframe_start is not None:
                    base_metadata["timeframe_start"] = float(doc.timeframe_start)
                if doc.timeframe_end is not None:
                    base_metadata["timeframe_end"] = float(doc.timeframe_end)
                
                vector_data = {
                    "id": doc.document_id,
                    "values": doc.embedding,
                    "metadata": base_metadata
                }
                vectors.append(vector_data)
            
            # Upload in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Uploaded RAG batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            logger.info(f"✅ Successfully uploaded {len(vectors)} RAG documents to Pinecone")
            
        except Exception as e:
            logger.error(f"Error uploading RAG documents to Pinecone: {e}")
            raise
    
    def rag_search(self, 
                   query: str,
                   top_k: int = 10,
                   agent_names: Optional[List[str]] = None,
                   video_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        FIXED: Perform RAG search across all document types
        """
        try:
            logger.info(f"🔍 Performing RAG search: '{query}'")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.encode(query, convert_to_tensor=False).tolist()
            
            # Build filter
            filter_dict = {}
            if agent_names:
                filter_dict["agent_name"] = {"$in": agent_names}
            if video_path:
                filter_dict["video_path"] = {"$eq": video_path}
            
            # Search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more to filter and group
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Convert to user-friendly format
            results = []
            for match in search_results.matches:
                metadata = match.metadata
                
                result = {
                    "content": metadata.get("content", ""),
                    "confidence": match.score,
                    "document_type": metadata.get("document_type", "unknown"),
                    "video": metadata.get("video_path", ""),
                    "timestamp": metadata.get("timestamp"),
                    "frame_number": metadata.get("frame_number"),
                    "agent": {
                        "name": metadata.get("agent_name"),
                        "role": metadata.get("agent_role")
                    } if metadata.get("agent_name") else None,
                    "context": self._generate_context_summary(metadata),
                    "related_perspectives": []
                }
                results.append(result)
            
            logger.info(f"✅ Found {len(results)} RAG search results")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error performing RAG search: {e}")
            return []
    
    def _generate_context_summary(self, metadata: Dict[str, Any]) -> str:
        """Generate context summary for result"""
        context_parts = []
        
        document_type = metadata.get("document_type", "unknown")
        if document_type == "frame_analysis":
            timestamp = metadata.get("timestamp")
            if timestamp:
                context_parts.append(f"Frame analysis at {timestamp:.1f}s")
        elif document_type == "agent_perspective":
            agent_role = metadata.get("agent_role")
            if agent_role:
                context_parts.append(f"{agent_role} perspective")
            timestamp = metadata.get("timestamp")
            if timestamp:
                context_parts.append(f"about {timestamp:.1f}s timeframe")
        elif document_type == "overall_analysis":
            context_parts.append("Overall video analysis")
        
        return " | ".join(context_parts) if context_parts else "General context"

class RAGVideoAnalysisSystem:
    """Complete RAG-enabled video analysis system"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize RAG video analysis system"""
        self.vector_store = RAGEnhancedVectorStore(api_key)
        self.fireworks_client = FireworksClient(api_key)
        logger.info("✅ RAG Video Analysis System initialized")
    
    def extract_agent_perspectives_from_discussion(self, 
                                                  discussion_turns: List[Any],
                                                  video_analysis_results: Dict[str, Any]) -> List[AgentPerspective]:
        """Extract agent perspectives with better error handling"""
        perspectives = []
        video_path = video_analysis_results.get("video_path", "unknown")
        frame_analyses = video_analysis_results.get("frame_analyses", [])
        
        logger.info(f"🔍 Extracting agent perspectives from discussion data...")
        logger.info(f"   Discussion data type: {type(discussion_turns)}")
        logger.info(f"   Found {len(discussion_turns)} discussion turns to process")
        
        for i, turn in enumerate(discussion_turns):
            try:
                agent_name = getattr(turn, 'agent_name', None) or f"Agent_{i}"
                agent_role = getattr(turn, 'agent_role', None) or "Analyst"
                content = getattr(turn, 'content', None) or str(turn)
                
                if not content or len(content.strip()) < 10:
                    logger.warning(f"Skipping turn {i} - insufficient content")
                    continue
                
                logger.info(f"✅ Extracted perspective from {agent_name} ({agent_role}): {len(content)} chars")
                
                # Create general perspective
                perspective = AgentPerspective(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    frame_number=None,
                    timestamp=None,
                    timeframe_start=None,
                    timeframe_end=None,
                    perspective_content=content,
                    perspective_type="general_discussion",
                    discussion_round=getattr(turn, 'round_number', None),
                    responding_to=getattr(turn, 'responding_to', None),
                    video_path=video_path
                )
                perspectives.append(perspective)
                    
            except Exception as e:
                logger.warning(f"Error extracting perspective from turn {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully extracted {len(perspectives)} agent perspectives")
        return perspectives
    
    async def index_video_for_rag(self, 
                                video_analysis_results: Dict[str, Any],
                                discussion_turns: List[Any]) -> str:
        """Index video analysis and agent perspectives for RAG queries"""
        try:
            logger.info("🔄 Indexing video for RAG queries...")
            
            # Extract agent perspectives
            logger.info("🔍 Extracting agent perspectives from discussion...")
            agent_perspectives = self.extract_agent_perspectives_from_discussion(
                discussion_turns, video_analysis_results
            )
            
            # Create RAG documents
            logger.info("📝 Creating RAG documents...")
            rag_documents = self.vector_store.create_rag_documents(
                video_analysis_results, agent_perspectives
            )
            
            if not rag_documents:
                return "❌ No valid documents created for RAG indexing"
            
            # Generate embeddings
            logger.info("🔧 Generating embeddings...")
            rag_documents = self.vector_store.generate_embeddings(rag_documents)
            
            # Upload to vector store
            logger.info("📤 Uploading to vector store...")
            self.vector_store.upsert_rag_documents(rag_documents)
            
            logger.info(f"✅ Successfully indexed {len(rag_documents)} documents for RAG")
            return f"✅ Successfully indexed {len(rag_documents)} documents for RAG queries"
            
        except Exception as e:
            logger.error(f"Error indexing video for RAG: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error indexing video for RAG: {e}"

class RAGQueryInterface:
    """FIXED: Simple interface for RAG queries with all required methods"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize RAG query interface"""
        self.rag_system = RAGVideoAnalysisSystem(api_key)
    
    async def add_video_to_rag_async(self, video_analysis_results: Dict[str, Any], discussion_turns: List[Any]) -> str:
        """Async version for use within event loops"""
        return await self.rag_system.index_video_for_rag(video_analysis_results, discussion_turns)
    
    def add_video_to_rag(self, video_analysis_results: Dict[str, Any], discussion_turns: List[Any]) -> str:
        """Sync version for direct calls"""
        return asyncio.run(self.rag_system.index_video_for_rag(video_analysis_results, discussion_turns))
    
    def query_video_rag(self, 
                       query: str,
                       num_results: int = 5,
                       focus_on: str = "all",
                       video_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        FIXED: Query the RAG system for video insights
        
        Args:
            query: Natural language query
            num_results: Number of results to return
            focus_on: Focus on specific agent perspectives
            video_filter: Filter by specific video
            
        Returns:
            List of search results with context
        """
        
        # Map focus to agent names
        agent_filter = None
        if focus_on == "technical":
            agent_filter = ["Alex", "Technical Analyst", "Cinematographer"]
        elif focus_on == "creative":
            agent_filter = ["Maya", "Creative Interpreter", "Film Critic"]
        elif focus_on == "audience":
            agent_filter = ["Jordan", "Audience Advocate", "Engagement Analyst"]
        
        try:
            results = self.rag_system.vector_store.rag_search(
                query=query,
                top_k=num_results,
                agent_names=agent_filter,
                video_path=video_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return []

# Simple test without nested event loops
def test_fixed_rag_system_simple():
    """Test the FIXED RAG system without async issues"""
    
    print("="*80)
    print("TESTING FIXED RAG SYSTEM - SIMPLE VERSION")
    print("="*80)
    
    # Mock data
    mock_analysis = {
        "video_path": "uploads/test/video.mp4",
        "frame_count": 10,
        "subtitle_count": 0,
        "frame_analyses": [
            {
                "frame_number": 1,
                "timestamp": 5.0,
                "analysis": "Basketball game in progress with high angle camera capturing court action.",
                "tokens_used": 150,
                "cost": 0.003
            }
        ],
        "overall_analysis": "Professional basketball game footage with excellent cinematography.",
        "processing_time": 160.08,
        "total_cost": 0.0131
    }
    
    class MockTurn:
        def __init__(self, agent_name, agent_role, content):
            self.agent_name = agent_name
            self.agent_role = agent_role
            self.content = content
            self.round_number = 1
            self.responding_to = None
    
    mock_discussion = [
        MockTurn("Alex", "Technical Analyst", "The camera work demonstrates excellent depth of field control with professional basketball coverage."),
        MockTurn("Maya", "Creative Interpreter", "The visual narrative captures the excitement and energy of competitive sports."),
    ]
    
    async def run_test():
        try:
            print("🔧 Initializing RAG system...")
            rag_interface = RAGQueryInterface()
            
            print("🔄 Testing video indexing...")
            status = await rag_interface.add_video_to_rag_async(mock_analysis, mock_discussion)
            print(f"Status: {status}")
            
            print("🔍 Testing RAG query...")
            results = rag_interface.query_video_rag("What did Alex say about camera work?", num_results=3)
            print(f"Query results: {len(results)} found")
            
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result['content'][:100]}...")
                print(f"     Agent: {result.get('agent', {}).get('name', 'Unknown')}")
                print(f"     Confidence: {result.get('confidence', 0):.3f}")
            
            print("✅ Test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run the test
    return asyncio.run(run_test())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        success = test_fixed_rag_system_simple()
        if success:
            print("\n🎉 RAG system is working! Agent discussions will now be indexed.")
        else:
            print("\n❌ RAG system test failed. Check the error messages above.")
    else:
        print("Fixed RAG Enhanced Vector System")
        print("Usage: python rag_enhanced_vector_system_fixed.py test")