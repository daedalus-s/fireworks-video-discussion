"""
Universal Temporal Agent Query Processor
Handles temporal queries for any video content with any configured agents
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TemporalQueryResult:
    """Result of a temporal query"""
    query_type: str
    extracted_elements: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    temporal_context: Dict[str, Any]
    response_text: str

class UniversalTemporalAgentQueryProcessor:
    """Universal temporal query processor for any video content and agents"""
    
    def __init__(self, rag_interface):
        """Initialize with RAG interface"""
        self.rag_interface = rag_interface
        
    def process_temporal_query(self, query: str) -> TemporalQueryResult:
        """Process temporal query and return structured result"""
        try:
            # Extract temporal elements
            extracted = self._extract_temporal_elements(query)
            
            # Determine query type
            query_type = self._classify_query_type(query, extracted)
            
            # Search based on query type
            search_results = self._perform_temporal_search(query, extracted, query_type)
            
            # Build temporal context
            temporal_context = self._build_temporal_context(search_results, extracted)
            
            # Generate response text
            response_text = self._generate_response_text(query, search_results, temporal_context, query_type)
            
            return TemporalQueryResult(
                query_type=query_type,
                extracted_elements=extracted,
                search_results=search_results,
                temporal_context=temporal_context,
                response_text=response_text
            )
            
        except Exception as e:
            logger.error(f"Temporal query processing failed: {e}")
            return TemporalQueryResult(
                query_type="error",
                extracted_elements={},
                search_results=[],
                temporal_context={},
                response_text=f"❌ Query processing failed: {e}"
            )
    
    def _extract_temporal_elements(self, query: str) -> Dict[str, Any]:
        """Extract temporal elements from query"""
        elements = {
            "frame_numbers": [],
            "timestamps": [],
            "agent_names": [],
            "when_queries": [],
            "content_targets": []
        }
        
        query_lower = query.lower()
        
        # Extract frame numbers
        frame_matches = re.findall(r'frame\s+(\d+)', query_lower)
        elements["frame_numbers"] = [int(f) for f in frame_matches]
        
        # Extract timestamps
        time_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b',
            r'(\d+):(\d+)',
            r'at\s+(\d+(?:\.\d+)?)\s*s'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:  # mm:ss format
                        time_sec = int(match[0]) * 60 + int(match[1])
                    else:
                        time_sec = float(match[0])
                else:
                    time_sec = float(match)
                elements["timestamps"].append(time_sec)
        
        # Extract agent names (common ones)
        agent_patterns = [
            r'\b(alex|maya|jordan|chen|isabella|marcus|taylor|casey|sam)\b',
            r'\b(cinematographer|critic|analyst|specialist|expert|director)\b'
        ]
        
        for pattern in agent_patterns:
            matches = re.findall(pattern, query_lower)
            elements["agent_names"].extend(matches)
        
        # Detect when queries
        when_patterns = [
            r'when\s+did\s+(.+?)\s+(?:happen|occur|start|begin|end|say|mention)',
            r'at\s+what\s+(?:time|frame)\s+(?:did|does)\s+(.+?)\s+(?:happen|occur)',
            r'what\s+time\s+(?:did|does)\s+(.+?)\s+(?:happen|occur|start)'
        ]
        
        for pattern in when_patterns:
            matches = re.findall(pattern, query_lower)
            elements["when_queries"].extend(matches)
        
        # Extract content targets
        content_patterns = [
            r'(?:what|describe|show|explain).*?(?:in\s+frame\s+\d+|at\s+\d+.*?s)',
            r'(?:lighting|camera|color|music|dialogue|character|scene|action)'
        ]
        
        for pattern in content_patterns:
            matches = re.findall(pattern, query_lower)
            elements["content_targets"].extend(matches)
        
        return elements
    
    def _classify_query_type(self, query: str, extracted: Dict[str, Any]) -> str:
        """Classify the type of temporal query"""
        query_lower = query.lower()
        
        if extracted["frame_numbers"]:
            return "frame_specific"
        elif extracted["timestamps"]:
            return "time_specific"
        elif extracted["when_queries"]:
            return "when_question"
        elif extracted["agent_names"]:
            return "agent_temporal"
        elif any(word in query_lower for word in ["when", "what time", "at what"]):
            return "temporal_search"
        else:
            return "general_temporal"
    
    def _perform_temporal_search(self, query: str, extracted: Dict[str, Any], query_type: str) -> List[Dict[str, Any]]:
        """Perform search based on query type"""
        try:
            # Use RAG interface for search
            if query_type == "agent_temporal" and extracted["agent_names"]:
                # Focus search on specific agents
                agent_focus = "technical" if "alex" in extracted["agent_names"] else "all"
                agent_focus = "creative" if "maya" in extracted["agent_names"] else agent_focus
                agent_focus = "audience" if "jordan" in extracted["agent_names"] else agent_focus
                
                results = self.rag_interface.query_video_rag(
                    query=query,
                    num_results=8,
                    focus_on=agent_focus
                )
            else:
                # General search
                results = self.rag_interface.query_video_rag(
                    query=query,
                    num_results=6
                )
            
            # Filter results based on temporal criteria
            filtered_results = self._filter_temporal_results(results, extracted)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            return []
    
    def _filter_temporal_results(self, results: List[Dict[str, Any]], extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results based on temporal criteria"""
        filtered = []
        
        for result in results:
            # Check frame number match
            if extracted["frame_numbers"] and result.get("frame_number"):
                if result["frame_number"] in extracted["frame_numbers"]:
                    result["temporal_match_reason"] = f"Frame {result['frame_number']} match"
                    filtered.append(result)
                    continue
            
            # Check timestamp match
            if extracted["timestamps"] and result.get("timestamp"):
                for target_time in extracted["timestamps"]:
                    time_diff = abs(result["timestamp"] - target_time)
                    if time_diff <= 10.0:  # Within 10 seconds
                        result["temporal_match_reason"] = f"Time {result['timestamp']:.1f}s (±{time_diff:.1f}s)"
                        filtered.append(result)
                        break
            
            # Check agent name match
            if extracted["agent_names"] and result.get("agent"):
                agent_name_lower = result["agent"]["name"].lower() if result["agent"] else ""
                for target_agent in extracted["agent_names"]:
                    if target_agent in agent_name_lower:
                        result["temporal_match_reason"] = f"Agent {result['agent']['name']} match"
                        filtered.append(result)
                        break
            
            # Include high-confidence general matches
            if result.get("confidence", 0) > 0.8:
                result["temporal_match_reason"] = f"High relevance ({result['confidence']:.3f})"
                filtered.append(result)
        
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_filtered = []
        for result in filtered:
            result_id = f"{result.get('video', '')}-{result.get('timestamp', 0)}-{result.get('frame_number', 0)}"
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_filtered.append(result)
        
        return sorted(unique_filtered, key=lambda x: x.get("confidence", 0), reverse=True)
    
    def _build_temporal_context(self, results: List[Dict[str, Any]], extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Build temporal context from results"""
        context = {
            "total_matches": len(results),
            "frame_matches": 0,
            "time_matches": 0,
            "agent_matches": 0,
            "time_range": {"min": None, "max": None},
            "agents_found": set(),
            "frames_found": set()
        }
        
        timestamps = []
        
        for result in results:
            # Count match types
            if "Frame" in result.get("temporal_match_reason", ""):
                context["frame_matches"] += 1
            if "Time" in result.get("temporal_match_reason", ""):
                context["time_matches"] += 1
            if "Agent" in result.get("temporal_match_reason", ""):
                context["agent_matches"] += 1
            
            # Collect temporal data
            if result.get("timestamp"):
                timestamps.append(result["timestamp"])
            if result.get("frame_number"):
                context["frames_found"].add(result["frame_number"])
            if result.get("agent") and result["agent"].get("name"):
                context["agents_found"].add(result["agent"]["name"])
        
        # Calculate time range
        if timestamps:
            context["time_range"]["min"] = min(timestamps)
            context["time_range"]["max"] = max(timestamps)
        
        # Convert sets to lists for JSON serialization
        context["agents_found"] = list(context["agents_found"])
        context["frames_found"] = list(context["frames_found"])
        
        return context
    
    def _generate_response_text(self, query: str, results: List[Dict[str, Any]], 
                               context: Dict[str, Any], query_type: str) -> str:
        """Generate formatted response text"""
        if not results:
            return f"❌ No temporal matches found for: '{query}'\n\n💡 Try:\n  • Different time references\n  • Alternative agent names\n  • More general search terms"
        
        response = f"🎯 **Temporal Query Results for:** '{query}'\n"
        response += f"📊 **Found {context['total_matches']} matches**"
        
        if context["time_range"]["min"] is not None:
            response += f" (timespan: {context['time_range']['min']:.1f}s - {context['time_range']['max']:.1f}s)"
        
        response += "\n" + "="*60 + "\n"
        
        # Show results by category
        for i, result in enumerate(results[:5], 1):  # Limit to top 5
            response += f"\n**{i}. [{result['document_type'].upper()}]** "
            response += f"Confidence: {result['confidence']:.3f}\n"
            
            # Show temporal match reason
            if "temporal_match_reason" in result:
                response += f"🎯 **Match:** {result['temporal_match_reason']}\n"
            
            # Show agent info
            if result.get("agent"):
                response += f"🤖 **Agent:** {result['agent']['name']} ({result['agent']['role']})\n"
            
            # Show temporal info
            if result.get("timestamp"):
                response += f"⏰ **Time:** {result['timestamp']:.1f}s"
                if result.get("frame_number"):
                    response += f" (Frame {result['frame_number']})"
                response += "\n"
            
            # Show content preview
            content_preview = result.get("content", "")[:200]
            response += f"📝 **Content:** {content_preview}{'...' if len(result.get('content', '')) > 200 else ''}\n"
            
            # Show context
            if result.get("context"):
                response += f"🔍 **Context:** {result['context']}\n"
            
            response += "-" * 50 + "\n"
        
        # Add summary
        if context["agents_found"]:
            response += f"\n👥 **Agents mentioned:** {', '.join(context['agents_found'])}\n"
        
        if context["frames_found"]:
            response += f"🎬 **Frames referenced:** {', '.join(map(str, sorted(context['frames_found'])))}\n"
        
        return response

def format_universal_temporal_response(result: TemporalQueryResult) -> str:
    """Format temporal query result for display"""
    return result.response_text

# Test function
def test_temporal_processor():
    """Test the temporal processor"""
    print("🧪 Testing Universal Temporal Processor")
    print("="*50)
    
    # Mock RAG interface for testing
    class MockRAGInterface:
        def query_video_rag(self, query: str, num_results: int = 5, focus_on: str = "all"):
            # Return mock results
            return [
                {
                    "content": f"Mock analysis result for: {query}",
                    "confidence": 0.85,
                    "document_type": "frame_analysis",
                    "video": "test_video.mp4",
                    "timestamp": 15.0,
                    "frame_number": 3,
                    "agent": {"name": "Alex", "role": "Technical Analyst"},
                    "context": "Technical analysis context"
                }
            ]
    
    # Test processor
    processor = UniversalTemporalAgentQueryProcessor(MockRAGInterface())
    
    test_queries = [
        "What happens in frame 3?",
        "At 15 seconds, what did Alex analyze?",
        "When did Maya mention lighting?",
        "What occurs at 1:30?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        result = processor.process_temporal_query(query)
        print(f"Query type: {result.query_type}")
        print(f"Extracted: {result.extracted_elements}")
        print(f"Results: {len(result.search_results)}")
        print("-" * 30)

if __name__ == "__main__":
    test_temporal_processor()