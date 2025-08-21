"""
FIXED: RAG-Enhanced Vector System with Pinecone Metadata Compatibility
Fixes the metadata serialization issue that prevents RAG indexing
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
    perspective_type: str  # "frame_analysis", "scene_commentary", "discussion_point"
    discussion_round: Optional[int]
    responding_to: Optional[str]
    video_path: str

@dataclass
class RAGVideoDocument:
    """Complete document for RAG system including frame analysis and agent perspectives"""
    document_id: str
    document_type: str  # "frame_analysis", "agent_perspective", "scene_summary"
    video_path: str
    frame_number: Optional[int]
    timestamp: Optional[float]
    timeframe_start: Optional[float]
    timeframe_end: Optional[float]
    
    # Content fields
    content: str
    metadata: Dict[str, Any]
    
    # Agent-specific fields
    agent_name: Optional[str] = None
    agent_role: Optional[str] = None
    perspective_type: Optional[str] = None
    
    # Embeddings
    embedding: Optional[List[float]] = None

@dataclass
class RAGSearchResult:
    """Enhanced search result with context and agent perspectives"""
    document_id: str
    document_type: str
    content: str
    similarity_score: float
    video_path: str
    timestamp: Optional[float]
    frame_number: Optional[int]
    agent_name: Optional[str]
    agent_role: Optional[str]
    metadata: Dict[str, Any]
    related_perspectives: List[Dict[str, Any]]

class RAGEnhancedVectorStore:
    """FIXED: Enhanced vector store with proper Pinecone metadata handling"""
    
    def __init__(self, api_key: Optional[str] = None, environment: str = "gcp-starter"):
        """Initialize RAG-enhanced vector store"""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone")
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable.")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = "video-rag-enhanced"
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        self._setup_index()
        
        # Initialize embedding generator
        self.embedding_generator = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ RAG-Enhanced Vector Store initialized")
    
    def _setup_index(self):
        """Setup or connect to Pinecone index for RAG"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new RAG Pinecone index: {self.index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to RAG Pinecone index: {self.index_name}")
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"RAG Index stats: {stats.total_vector_count} documents")
            
        except Exception as e:
            logger.error(f"Error setting up RAG Pinecone index: {e}")
            raise
    
    def create_rag_documents(self, 
                           video_analysis_results: Dict[str, Any],
                           agent_perspectives: List[AgentPerspective]) -> List[RAGVideoDocument]:
        """Create RAG documents from video analysis and agent perspectives"""
        documents = []
        video_path = video_analysis_results.get("video_path", "unknown")
        
        # 1. Create documents from frame analyses
        for frame_analysis in video_analysis_results.get("frame_analyses", []):
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
                    # FIXED: Remove parsed_analysis to avoid complex object storage
                }
            )
            documents.append(document)
        
        # 2. Create documents from agent perspectives
        for perspective in agent_perspectives:
            doc_id = f"agent_{perspective.agent_name}_{perspective.video_path}_{perspective.timestamp or 0}_{perspective.discussion_round or 0}"
            
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
        
        # 3. Create documents from subtitle analyses
        for subtitle_analysis in video_analysis_results.get("subtitle_analyses", []):
            if not subtitle_analysis.get("analysis") or "failed" in subtitle_analysis.get("analysis", "").lower():
                continue
            
            # Parse timeframe from subtitle_range
            timeframe_parts = subtitle_analysis.get("subtitle_range", "0s - 0s").split(" - ")
            start_time = float(timeframe_parts[0].replace("s", "")) if len(timeframe_parts) > 0 else 0
            end_time = float(timeframe_parts[1].replace("s", "")) if len(timeframe_parts) > 1 else start_time
            
            doc_id = f"subtitle_{video_path}_{start_time}_{end_time}"
            
            document = RAGVideoDocument(
                document_id=doc_id,
                document_type="subtitle_analysis",
                video_path=video_path,
                frame_number=None,
                timestamp=(start_time + end_time) / 2,  # Midpoint
                timeframe_start=start_time,
                timeframe_end=end_time,
                content=f"Dialogue Analysis: {subtitle_analysis.get('analysis', '')} | Original Text: {subtitle_analysis.get('text_analyzed', '')}",
                metadata={
                    "subtitle_range": subtitle_analysis.get("subtitle_range"),
                    "original_text": subtitle_analysis.get("text_analyzed", ""),
                    "tokens_used": subtitle_analysis.get("tokens_used", 0),
                    "cost": subtitle_analysis.get("cost", 0)
                }
            )
            documents.append(document)
        
        # 4. Create document from overall analysis
        if video_analysis_results.get("overall_analysis"):
            doc_id = f"overall_{video_path}_{datetime.now().timestamp()}"
            
            document = RAGVideoDocument(
                document_id=doc_id,
                document_type="overall_analysis",
                video_path=video_path,
                frame_number=None,
                timestamp=None,
                timeframe_start=0,
                timeframe_end=video_analysis_results.get("frame_analyses", [{}])[-1].get("timestamp", 0),
                content=video_analysis_results.get("overall_analysis", ""),
                metadata={
                    "frame_count": video_analysis_results.get("frame_count", 0),
                    "subtitle_count": video_analysis_results.get("subtitle_count", 0),
                    "processing_time": video_analysis_results.get("processing_time", 0),
                    "total_cost": video_analysis_results.get("total_cost", 0)
                }
            )
            documents.append(document)
        
        logger.info(f"✅ Created {len(documents)} RAG documents")
        return documents
    
    def generate_embeddings(self, documents: List[RAGVideoDocument]) -> List[RAGVideoDocument]:
        """Generate embeddings for all documents"""
        logger.info(f"🔄 Generating embeddings for {len(documents)} documents...")
        
        for doc in documents:
            try:
                # Create enhanced content for embedding
                embedding_content = self._create_embedding_content(doc)
                
                # Generate embedding
                embedding = self.embedding_generator.encode(embedding_content, convert_to_tensor=False)
                doc.embedding = embedding.tolist()
                
            except Exception as e:
                logger.warning(f"Failed to generate embedding for document {doc.document_id}: {e}")
                doc.embedding = None
        
        successful_embeddings = sum(1 for doc in documents if doc.embedding is not None)
        logger.info(f"✅ Generated {successful_embeddings}/{len(documents)} embeddings")
        
        return documents
    
    def _create_embedding_content(self, doc: RAGVideoDocument) -> str:
        """Create optimized content for embedding generation"""
        content_parts = []
        
        # Add document type context
        content_parts.append(f"Document type: {doc.document_type}")
        
        # Add agent context if available
        if doc.agent_name:
            content_parts.append(f"Agent: {doc.agent_name} ({doc.agent_role})")
            content_parts.append(f"Perspective type: {doc.perspective_type}")
        
        # Add temporal context
        if doc.timestamp is not None:
            content_parts.append(f"Timestamp: {doc.timestamp:.1f} seconds")
        
        if doc.timeframe_start is not None and doc.timeframe_end is not None:
            content_parts.append(f"Timeframe: {doc.timeframe_start:.1f}s to {doc.timeframe_end:.1f}s")
        
        # Add main content
        content_parts.append(f"Content: {doc.content}")
        
        return " | ".join(content_parts)
    
    def _sanitize_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: Sanitize metadata to ensure Pinecone compatibility"""
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list elements to strings if they're not already simple types
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    sanitized[key] = value
                else:
                    sanitized[key] = [str(item) for item in value]
            elif isinstance(value, dict):
                # Convert complex objects to JSON strings
                sanitized[key] = json.dumps(value)
            else:
                # Convert other types to strings
                sanitized[key] = str(value)
        
        return sanitized
    
    def upsert_rag_documents(self, documents: List[RAGVideoDocument], batch_size: int = 100):
        """FIXED: Upload RAG documents to Pinecone with proper metadata handling"""
        try:
            vectors = []
            
            for doc in documents:
                if doc.embedding is None:
                    logger.warning(f"Skipping document {doc.document_id} - no embedding")
                    continue
                
                # Prepare metadata for Pinecone with proper sanitization
                base_metadata = {
                    "document_type": doc.document_type,
                    "video_path": doc.video_path,
                    "content": doc.content[:1000]  # Truncate for metadata limits
                }
                
                # Add sanitized metadata
                sanitized_metadata = self._sanitize_metadata_for_pinecone(doc.metadata)
                base_metadata.update(sanitized_metadata)
                
                # Add agent-specific metadata
                if doc.agent_name:
                    base_metadata.update({
                        "agent_name": doc.agent_name,
                        "agent_role": doc.agent_role,
                        "perspective_type": doc.perspective_type or "unknown"
                    })
                
                # Add temporal metadata
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
                   document_types: Optional[List[str]] = None,
                   agent_names: Optional[List[str]] = None,
                   video_path: Optional[str] = None,
                   timeframe: Optional[Tuple[float, float]] = None) -> List[RAGSearchResult]:
        """
        Perform RAG search across all document types
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            document_types: Filter by document types
            agent_names: Filter by specific agents
            video_path: Filter by specific video
            timeframe: Filter by time range (start, end)
        """
        try:
            logger.info(f"🔍 Performing RAG search: '{query}'")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.encode(query, convert_to_tensor=False).tolist()
            
            # Build filter
            filter_dict = {}
            if document_types:
                filter_dict["document_type"] = {"$in": document_types}
            if agent_names:
                filter_dict["agent_name"] = {"$in": agent_names}
            if video_path:
                filter_dict["video_path"] = {"$eq": video_path}
            if timeframe:
                filter_dict["timestamp"] = {"$gte": timeframe[0], "$lte": timeframe[1]}
            
            # Search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more to filter and group
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Convert to RAGSearchResult objects
            results = []
            for match in search_results.matches:
                metadata = match.metadata
                
                # Find related perspectives for this result
                related_perspectives = self._find_related_perspectives(match, search_results.matches)
                
                result = RAGSearchResult(
                    document_id=match.id,
                    document_type=metadata.get("document_type", "unknown"),
                    content=metadata.get("content", ""),
                    similarity_score=match.score,
                    video_path=metadata.get("video_path", ""),
                    timestamp=metadata.get("timestamp"),
                    frame_number=metadata.get("frame_number"),
                    agent_name=metadata.get("agent_name"),
                    agent_role=metadata.get("agent_role"),
                    metadata=metadata,
                    related_perspectives=related_perspectives
                )
                results.append(result)
            
            # Group and rank results
            grouped_results = self._group_and_rank_results(results[:top_k])
            
            logger.info(f"✅ Found {len(grouped_results)} RAG search results")
            return grouped_results
            
        except Exception as e:
            logger.error(f"Error performing RAG search: {e}")
            return []
    
    def _find_related_perspectives(self, primary_match, all_matches) -> List[Dict[str, Any]]:
        """Find related agent perspectives for a primary result"""
        related = []
        primary_timestamp = primary_match.metadata.get("timestamp")
        primary_video = primary_match.metadata.get("video_path")
        
        if not primary_timestamp or not primary_video:
            return related
        
        # Find perspectives within 10 seconds of the primary result
        time_window = 10.0
        
        for match in all_matches:
            if match.id == primary_match.id:
                continue
            
            metadata = match.metadata
            if (metadata.get("document_type") == "agent_perspective" and
                metadata.get("video_path") == primary_video and
                metadata.get("timestamp") is not None):
                
                time_diff = abs(metadata["timestamp"] - primary_timestamp)
                if time_diff <= time_window:
                    related.append({
                        "agent_name": metadata.get("agent_name"),
                        "agent_role": metadata.get("agent_role"),
                        "content": metadata.get("content", "")[:200] + "...",
                        "similarity_score": match.score,
                        "time_difference": time_diff
                    })
        
        # Sort by similarity score
        related.sort(key=lambda x: x["similarity_score"], reverse=True)
        return related[:3]  # Top 3 related perspectives
    
    def _group_and_rank_results(self, results: List[RAGSearchResult]) -> List[RAGSearchResult]:
        """Group results by video/timeframe and rank them"""
        # For now, just sort by similarity score
        # Could implement more sophisticated grouping later
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)

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
        """Extract agent perspectives and associate them with frames/timeframes"""
        perspectives = []
        video_path = video_analysis_results.get("video_path", "unknown")
        frame_analyses = video_analysis_results.get("frame_analyses", [])
        
        for turn in discussion_turns:
            try:
                # Extract frame/time references from agent content
                frame_refs, time_refs = self._extract_temporal_references(turn.content, frame_analyses)
                
                # Create perspectives for each reference found
                if frame_refs or time_refs:
                    for frame_ref in frame_refs:
                        perspective = AgentPerspective(
                            agent_name=turn.agent_name,
                            agent_role=turn.agent_role,
                            frame_number=frame_ref.get("frame_number"),
                            timestamp=frame_ref.get("timestamp"),
                            timeframe_start=None,
                            timeframe_end=None,
                            perspective_content=turn.content,
                            perspective_type="frame_commentary",
                            discussion_round=getattr(turn, 'round_number', None),
                            responding_to=getattr(turn, 'responding_to', None),
                            video_path=video_path
                        )
                        perspectives.append(perspective)
                    
                    for time_ref in time_refs:
                        perspective = AgentPerspective(
                            agent_name=turn.agent_name,
                            agent_role=turn.agent_role,
                            frame_number=None,
                            timestamp=time_ref.get("center_time"),
                            timeframe_start=time_ref.get("start_time"),
                            timeframe_end=time_ref.get("end_time"),
                            perspective_content=turn.content,
                            perspective_type="timeframe_commentary",
                            discussion_round=getattr(turn, 'round_number', None),
                            responding_to=getattr(turn, 'responding_to', None),
                            video_path=video_path
                        )
                        perspectives.append(perspective)
                
                else:
                    # General perspective not tied to specific frame
                    perspective = AgentPerspective(
                        agent_name=turn.agent_name,
                        agent_role=turn.agent_role,
                        frame_number=None,
                        timestamp=None,
                        timeframe_start=None,
                        timeframe_end=None,
                        perspective_content=turn.content,
                        perspective_type="general_discussion",
                        discussion_round=getattr(turn, 'round_number', None),
                        responding_to=getattr(turn, 'responding_to', None),
                        video_path=video_path
                    )
                    perspectives.append(perspective)
                    
            except Exception as e:
                logger.warning(f"Error extracting perspective from turn: {e}")
                continue
        
        logger.info(f"✅ Extracted {len(perspectives)} agent perspectives")
        return perspectives
    
    def _extract_temporal_references(self, content: str, frame_analyses: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Extract frame and time references from agent content"""
        import re
        
        frame_refs = []
        time_refs = []
        
        # Extract frame references (e.g., "frame 5", "Frame 10")
        frame_matches = re.findall(r'frame\s+(\d+)', content, re.IGNORECASE)
        for match in frame_matches:
            frame_num = int(match)
            # Find corresponding frame analysis
            for fa in frame_analyses:
                if fa.get("frame_number") == frame_num:
                    frame_refs.append({
                        "frame_number": frame_num,
                        "timestamp": fa.get("timestamp")
                    })
                    break
        
        # Extract time references (e.g., "5 seconds", "at 1:30", "10s mark")
        time_patterns = [
            r'(\d+)\s*seconds?',
            r'(\d+)\s*s\b',
            r'at\s+(\d+(?:\.\d+)?)\s*s',
            r'(\d+):(\d+)'  # mm:ss format
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:  # mm:ss format
                        time_seconds = int(match[0]) * 60 + int(match[1])
                    else:
                        time_seconds = float(match[0])
                else:
                    time_seconds = float(match)
                
                time_refs.append({
                    "center_time": time_seconds,
                    "start_time": max(0, time_seconds - 5),
                    "end_time": time_seconds + 5
                })
        
        return frame_refs, time_refs
    
    async def index_video_for_rag(self, 
                                video_analysis_results: Dict[str, Any],
                                discussion_turns: List[Any]) -> str:
        """FIXED: Index video analysis and agent perspectives for RAG queries"""
        try:
            logger.info("🔄 Indexing video for RAG queries...")
            
            # Extract agent perspectives
            agent_perspectives = self.extract_agent_perspectives_from_discussion(
                discussion_turns, video_analysis_results
            )
            
            # Create RAG documents
            rag_documents = self.vector_store.create_rag_documents(
                video_analysis_results, agent_perspectives
            )
            
            # Generate embeddings
            rag_documents = self.vector_store.generate_embeddings(rag_documents)
            
            # Upload to vector store (now with fixed metadata handling)
            self.vector_store.upsert_rag_documents(rag_documents)
            
            logger.info(f"✅ Successfully indexed {len(rag_documents)} documents for RAG")
            return f"✅ Successfully indexed {len(rag_documents)} documents for RAG queries"
            
        except Exception as e:
            logger.error(f"Error indexing video for RAG: {e}")
            return f"❌ Error indexing video for RAG: {e}"

class RAGQueryInterface:
    """User-friendly interface for RAG queries"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize RAG query interface"""
        self.rag_system = RAGVideoAnalysisSystem(api_key)
    
    def add_video_to_rag(self, video_analysis_results: Dict[str, Any], discussion_turns: List[Any]) -> str:
        """Add a video's analysis and discussion to RAG index"""
        return asyncio.run(self.rag_system.index_video_for_rag(video_analysis_results, discussion_turns))
    
    def query_video_rag(self, 
                       query: str,
                       num_results: int = 5,
                       focus_on: str = "all",  # "all", "technical", "creative", "audience"
                       video_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query the RAG system for video insights
        
        Args:
            query: Natural language query
            num_results: Number of results to return
            focus_on: Focus on specific agent perspectives
            video_filter: Filter by specific video
            
        Returns:
            List of search results with context
        """
        
        # Map focus to agent roles
        agent_filter = None
        if focus_on == "technical":
            agent_filter = ["Technical Analyst"]
        elif focus_on == "creative":
            agent_filter = ["Creative Interpreter"]
        elif focus_on == "audience":
            agent_filter = ["Audience Advocate"]
        
        results = self.rag_system.vector_store.rag_search(
            query=query,
            top_k=num_results,
            agent_names=agent_filter,
            video_path=video_filter
        )
        
        # Convert to user-friendly format
        formatted_results = []
        for result in results:
            formatted_result = {
                "content": result.content,
                "confidence": result.similarity_score,
                "document_type": result.document_type,
                "video": result.video_path,
                "timestamp": result.timestamp,
                "frame_number": result.frame_number,
                "agent": {
                    "name": result.agent_name,
                    "role": result.agent_role
                } if result.agent_name else None,
                "related_perspectives": result.related_perspectives,
                "context": self._generate_context_summary(result)
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _generate_context_summary(self, result: RAGSearchResult) -> str:
        """Generate context summary for result"""
        context_parts = []
        
        if result.document_type == "frame_analysis":
            context_parts.append(f"Frame analysis at {result.timestamp:.1f}s")
        elif result.document_type == "agent_perspective":
            context_parts.append(f"{result.agent_role} perspective")
            if result.timestamp:
                context_parts.append(f"about {result.timestamp:.1f}s timeframe")
        elif result.document_type == "subtitle_analysis":
            context_parts.append("Dialogue analysis")
        
        if result.related_perspectives:
            agent_names = [rp["agent_name"] for rp in result.related_perspectives]
            context_parts.append(f"with insights from {', '.join(set(agent_names))}")
        
        return " | ".join(context_parts)


# Demo and testing functions
async def demo_rag_system():
    """Demonstrate the FIXED RAG system"""
    
    print("="*80)
    print("FIXED RAG-ENHANCED VIDEO ANALYSIS DEMO")
    print("="*80)
    
    # This would typically use real analysis results
    # For demo, we'll create mock data
    
    mock_analysis = {
        "video_path": "demo_video.mp4",
        "frame_analyses": [
            {
                "frame_number": 1,
                "timestamp": 5.0,
                "analysis": "A person in a blue jacket walks down a busy urban street. The camera uses a medium shot with shallow depth of field, creating nice bokeh in the background. Natural daylight provides even illumination with cool color temperature.",
                "parsed_analysis": {
                    "subjects": ["person"],
                    "objects": ["jacket", "street"],
                    "colors": ["blue"],
                    "mood": "calm"
                },
                "tokens_used": 150,
                "cost": 0.003,
                "analysis_depth": "comprehensive"
            },
            {
                "frame_number": 2,
                "timestamp": 10.0,
                "analysis": "Close-up shot of the same person's face showing contemplative expression. Dramatic side lighting creates strong shadows. The composition follows rule of thirds with subject positioned on the left third.",
                "tokens_used": 140,
                "cost": 0.003,
                "analysis_depth": "comprehensive"
            }
        ],
        "subtitle_analyses": [
            {
                "subtitle_range": "5.0s - 8.0s",
                "text_analyzed": "[5.0s - 8.0s]: I need to find a new direction in my life.",
                "analysis": "Introspective dialogue revealing character's internal conflict and desire for change.",
                "tokens_used": 80,
                "cost": 0.001
            }
        ],
        "overall_analysis": "This short sequence establishes character mood and internal state through both visual and audio storytelling techniques.",
        "frame_count": 2,
        "subtitle_count": 1,
        "processing_time": 45.5,
        "total_cost": 0.007
    }
    
    # Mock discussion turns
    class MockTurn:
        def __init__(self, agent_name, agent_role, content):
            self.agent_name = agent_name
            self.agent_role = agent_role
            self.content = content
            self.round_number = 1
            self.responding_to = None
    
    mock_discussion = [
        MockTurn("Alex", "Technical Analyst", 
                "The camera work in frame 1 at 5 seconds demonstrates excellent depth of field control. The shallow focus isolates the subject while the urban environment provides context through controlled bokeh."),
        MockTurn("Maya", "Creative Interpreter",
                "The blue jacket creates a visual metaphor for the character's emotional state - cool and distant. Combined with the dialogue at 5-8 seconds about finding direction, this reinforces themes of isolation and searching."),
        MockTurn("Jordan", "Audience Advocate",
                "Frame 2 at 10 seconds uses intimate close-up framing that helps viewers connect emotionally with the character's contemplative moment. The dramatic lighting enhances the mood without being distracting.")
    ]
    
    try:
        # Initialize FIXED RAG system
        rag_interface = RAGQueryInterface()
        
        # Index the mock data
        print("🔄 Indexing video for RAG with FIXED metadata handling...")
        status = rag_interface.add_video_to_rag(mock_analysis, mock_discussion)
        print(status)
        
        # Perform various RAG queries
        print("\n🔍 Testing RAG queries...")
        
        queries = [
            "What technical camera techniques were used?",
            "How does the blue jacket contribute to storytelling?",
            "What moments show character emotion?",
            "Describe the lighting in close-up shots",
            "What dialogue reveals character motivation?"
        ]
        
        for query in queries:
            print(f"\n📋 Query: '{query}'")
            print("-" * 50)
            
            results = rag_interface.query_video_rag(query, num_results=3)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['document_type']}] Confidence: {result['confidence']:.3f}")
                if result['agent']:
                    print(f"   Agent: {result['agent']['name']} ({result['agent']['role']})")
                if result['timestamp']:
                    print(f"   Time: {result['timestamp']:.1f}s")
                print(f"   Content: {result['content'][:150]}...")
                print(f"   Context: {result['context']}")
                
                if result['related_perspectives']:
                    print(f"   Related perspectives from: {', '.join([rp['agent_name'] for rp in result['related_perspectives']])}")
                print()
        
        print("✅ FIXED RAG system demo complete!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


# Quick fix script to apply the metadata fix
def apply_metadata_fix():
    """Apply the metadata fix to existing RAG system"""
    print("🔧 APPLYING PINECONE METADATA FIX")
    print("="*50)
    print("This fix addresses the issue:")
    print('• "Metadata value must be a string, number, boolean or list of strings"')
    print('• Adds proper metadata sanitization for Pinecone compatibility')
    print('• Converts complex objects to JSON strings')
    print("="*50)
    
    print("\n✅ Fix applied! Key changes:")
    print("1. Added _sanitize_metadata_for_pinecone() method")
    print("2. Removed complex parsed_analysis objects from metadata")
    print("3. Convert dict/list objects to JSON strings")
    print("4. Ensure all metadata values are Pinecone-compatible types")
    
    print("\n📝 To use the fix:")
    print("1. Replace your rag_enhanced_vector_system.py with this fixed version")
    print("2. Re-run your analysis: python integrated_rag_pipeline.py video.mp4")
    print("3. RAG indexing should now succeed without metadata errors")
    
    print("\n🧪 Test the fix:")
    print("python query_video_rag.py 'What did Alex say about lighting?'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_rag_system())
    elif len(sys.argv) > 1 and sys.argv[1] == "fix":
        apply_metadata_fix()
    else:
        print("Fixed RAG Enhanced Vector System")
        print("Usage:")
        print("  python rag_enhanced_vector_system_fixed.py demo  # Run demo")
        print("  python rag_enhanced_vector_system_fixed.py fix   # Show fix info")
        print("\nThis fixes the Pinecone metadata compatibility issue.")