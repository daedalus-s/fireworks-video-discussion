"""
Enhanced Temporal Agent Query System
Enables precise queries about agent perspectives at specific times/frames
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from rag_enhanced_vector_system import RAGQueryInterface

class TemporalAgentQueryProcessor:
    """Processes complex temporal and agent-specific queries"""
    
    def __init__(self, rag_interface: RAGQueryInterface):
        self.rag_interface = rag_interface
        
        # Define query patterns for different types of temporal agent queries
        self.query_patterns = {
            "agent_time_concept": [
                r"(?:at what|when did|what frame|what time).*?(\w+).*?(?:think|say|mention|discuss|interpret).*?(\w+.*?)(?:\?|$)",
                r"(\w+).*?(?:said|thought|mentioned).*?(\w+.*?)(?:at|during|in).*?(?:frame|time|second)",
                r"(?:when|what frame).*?(\w+).*?(?:interpreted|analyzed|discussed).*?(\w+.*?)(?:\?|$)"
            ],
            "concept_agent_time": [
                r"(?:when|what frame|what time).*?(\w+.*?).*?(?:according to|by|from).*?(\w+)(?:\?|$)",
                r"(?:at what point|when).*?(\w+.*?).*?(\w+)(?:'s|s)?.*?(?:perspective|view|opinion)(?:\?|$)"
            ],
            "frame_agent_concept": [
                r"(?:in frame|at frame|frame).*?(\d+).*?(\w+).*?(?:said|thought|discussed).*?(\w+.*?)(?:\?|$)",
                r"(?:at).*?(\d+(?:\.\d+)?)\s*(?:s|seconds?).*?(\w+).*?(?:mentioned|analyzed).*?(\w+.*?)(?:\?|$)"
            ]
        }
        
        # Agent name mappings
        self.agent_mappings = {
            "alex": "Alex",
            "maya": "Maya", 
            "jordan": "Jordan",
            "technical": "Alex",
            "creative": "Maya",
            "audience": "Jordan",
            "analyst": "Alex",
            "interpreter": "Maya",
            "advocate": "Jordan"
        }
    
    def process_temporal_agent_query(self, query: str) -> Dict[str, Any]:
        """
        Process complex temporal agent queries and return structured search
        
        Examples of queries this handles:
        - "At what frame did Maya think that blue symbolized peace?"
        - "When did Alex mention dramatic lighting?"
        - "What time did Jordan discuss audience engagement?"
        - "In frame 5, what did Maya say about color symbolism?"
        """
        
        query_lower = query.lower()
        
        # Extract components from the query
        extracted = self._extract_query_components(query_lower)
        
        if not extracted:
            # Fallback to general search
            return self._perform_general_search(query)
        
        # Perform targeted search based on extracted components
        return self._perform_targeted_search(query, extracted)
    
    def _extract_query_components(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract agent, concept, and temporal components from query"""
        
        components = {
            "agent": None,
            "concept": None,
            "temporal_info": None,
            "query_type": None
        }
        
        # Try each pattern type
        for pattern_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    components = self._parse_match(match, pattern_type)
                    components["query_type"] = pattern_type
                    if components["agent"] or components["concept"]:
                        return components
        
        return None
    
    def _parse_match(self, match, pattern_type: str) -> Dict[str, Any]:
        """Parse regex match based on pattern type"""
        
        components = {"agent": None, "concept": None, "temporal_info": None}
        
        if pattern_type == "agent_time_concept":
            # Pattern: "When did Maya think blue symbolized peace?"
            potential_agent = match.group(1).lower()
            concept = match.group(2).strip()
            
            components["agent"] = self._normalize_agent_name(potential_agent)
            components["concept"] = concept
            
        elif pattern_type == "concept_agent_time":
            # Pattern: "When was peace mentioned by Maya?"
            concept = match.group(1).strip()
            potential_agent = match.group(2).lower()
            
            components["concept"] = concept
            components["agent"] = self._normalize_agent_name(potential_agent)
            
        elif pattern_type == "frame_agent_concept":
            # Pattern: "In frame 5, what did Maya say about peace?"
            temporal = match.group(1)
            potential_agent = match.group(2).lower()
            concept = match.group(3).strip() if len(match.groups()) > 2 else None
            
            components["temporal_info"] = temporal
            components["agent"] = self._normalize_agent_name(potential_agent)
            components["concept"] = concept
        
        return components
    
    def _normalize_agent_name(self, name: str) -> Optional[str]:
        """Normalize agent name variations"""
        name = name.lower().strip()
        return self.agent_mappings.get(name)
    
    def _perform_targeted_search(self, original_query: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """Perform targeted search based on extracted components"""
        
        # Build enhanced query
        query_parts = []
        
        if components["concept"]:
            query_parts.append(components["concept"])
        
        # Add agent-specific context
        if components["agent"]:
            agent_context = self._get_agent_context(components["agent"])
            query_parts.append(agent_context)
        
        enhanced_query = " ".join(query_parts)
        
        # Perform search with filters
        search_kwargs = {
            "query": enhanced_query,
            "num_results": 10,  # Get more results for temporal filtering
        }
        
        if components["agent"]:
            search_kwargs["focus_on"] = self._agent_to_focus(components["agent"])
        
        # Perform initial search
        results = self.rag_interface.query_video_rag(**search_kwargs)
        
        # Post-process results for temporal information
        filtered_results = self._filter_temporal_results(results, components)
        
        return {
            "original_query": original_query,
            "components": components,
            "enhanced_query": enhanced_query,
            "results": filtered_results,
            "temporal_answers": self._extract_temporal_answers(filtered_results, components)
        }
    
    def _get_agent_context(self, agent: str) -> str:
        """Get contextual terms for each agent to improve search"""
        contexts = {
            "Alex": "technical camera lighting cinematography production",
            "Maya": "creative artistic symbolism metaphor emotion storytelling",
            "Jordan": "audience engagement viewer experience accessibility"
        }
        return contexts.get(agent, "")
    
    def _agent_to_focus(self, agent: str) -> str:
        """Convert agent name to focus parameter"""
        focus_map = {
            "Alex": "technical",
            "Maya": "creative", 
            "Jordan": "audience"
        }
        return focus_map.get(agent, "all")
    
    def _filter_temporal_results(self, results: List[Dict], components: Dict[str, Any]) -> List[Dict]:
        """Filter results based on temporal requirements"""
        
        if not components.get("temporal_info"):
            return results
        
        # If specific frame/time mentioned, prioritize those results
        temporal_value = components["temporal_info"]
        
        try:
            if temporal_value.isdigit():
                # Frame number
                frame_num = int(temporal_value)
                filtered = [r for r in results if r.get("frame_number") == frame_num]
                if filtered:
                    return filtered
            else:
                # Time value
                time_val = float(temporal_value)
                # Find results within 5 seconds
                filtered = [r for r in results 
                           if r.get("timestamp") and abs(r["timestamp"] - time_val) <= 5.0]
                if filtered:
                    return filtered
        except ValueError:
            pass
        
        return results
    
    def _extract_temporal_answers(self, results: List[Dict], components: Dict[str, Any]) -> List[Dict]:
        """Extract specific temporal answers from results"""
        
        answers = []
        
        for result in results:
            if result.get("agent") and result["agent"]["name"] == components.get("agent"):
                
                answer = {
                    "frame_number": result.get("frame_number"),
                    "timestamp": result.get("timestamp"),
                    "agent": result["agent"]["name"],
                    "agent_role": result["agent"]["role"],
                    "content": result["content"],
                    "confidence": result["confidence"],
                    "video": result["video"]
                }
                
                # Extract specific mentions of the concept
                if components.get("concept"):
                    concept_mentions = self._find_concept_mentions(
                        result["content"], 
                        components["concept"]
                    )
                    answer["concept_mentions"] = concept_mentions
                
                answers.append(answer)
        
        # Sort by relevance (confidence and temporal closeness)
        answers.sort(key=lambda x: x["confidence"], reverse=True)
        
        return answers
    
    def _find_concept_mentions(self, content: str, concept: str) -> List[str]:
        """Find specific mentions of concept in content"""
        
        content_lower = content.lower()
        concept_lower = concept.lower()
        
        mentions = []
        
        # Find sentences containing the concept
        sentences = content.split('.')
        for sentence in sentences:
            if concept_lower in sentence.lower():
                mentions.append(sentence.strip())
        
        return mentions
    
    def _perform_general_search(self, query: str) -> Dict[str, Any]:
        """Fallback to general search if pattern matching fails"""
        
        results = self.rag_interface.query_video_rag(query, num_results=5)
        
        return {
            "original_query": query,
            "components": {"extracted": False},
            "enhanced_query": query,
            "results": results,
            "temporal_answers": []
        }

def format_temporal_response(query_result: Dict[str, Any]) -> str:
    """Format the response for temporal agent queries"""
    
    response_parts = []
    original_query = query_result["original_query"]
    components = query_result["components"]
    temporal_answers = query_result["temporal_answers"]
    
    response_parts.append(f"🔍 Query: {original_query}")
    response_parts.append("="*60)
    
    if not temporal_answers:
        response_parts.append("❌ No specific temporal matches found.")
        
        # Show general results if available
        if query_result["results"]:
            response_parts.append("\n💡 Related results found:")
            for i, result in enumerate(query_result["results"][:3], 1):
                response_parts.append(f"{i}. {result.get('agent', {}).get('name', 'Unknown')} at {result.get('timestamp', 'unknown')}s:")
                response_parts.append(f"   {result['content'][:150]}...")
        
        return "\n".join(response_parts)
    
    # Show temporal answers
    agent_name = components.get("agent")
    concept = components.get("concept")
    
    if agent_name and concept:
        response_parts.append(f"📍 Found {len(temporal_answers)} instances where {agent_name} discussed '{concept}':")
    else:
        response_parts.append(f"📍 Found {len(temporal_answers)} relevant temporal matches:")
    
    response_parts.append("")
    
    for i, answer in enumerate(temporal_answers, 1):
        
        # Temporal information
        temporal_info = []
        if answer.get("frame_number"):
            temporal_info.append(f"Frame {answer['frame_number']}")
        if answer.get("timestamp"):
            temporal_info.append(f"{answer['timestamp']:.1f}s")
        
        temporal_str = " (".join(temporal_info) + ")" if temporal_info else ""
        
        response_parts.append(f"{i}. 🎯 {answer['agent']} ({answer['agent_role']}){temporal_str}")
        response_parts.append(f"   📊 Confidence: {answer['confidence']:.3f}")
        response_parts.append(f"   🎬 Video: {answer['video']}")
        
        # Show concept mentions if available
        if answer.get("concept_mentions"):
            response_parts.append(f"   💭 Relevant quote:")
            for mention in answer["concept_mentions"][:1]:  # Show first mention
                response_parts.append(f"      \"{mention.strip()}\"")
        else:
            response_parts.append(f"   💭 Content: {answer['content'][:200]}...")
        
        response_parts.append("")
    
    return "\n".join(response_parts)

# Enhanced query examples and test function
def demonstrate_temporal_queries():
    """Demonstrate various temporal agent queries"""
    
    print("🧠 TEMPORAL AGENT QUERY DEMONSTRATION")
    print("="*60)
    
    # Example queries that the system can handle
    example_queries = [
        "At what frame did Maya think that blue symbolized peace?",
        "When did Alex mention dramatic lighting?",
        "What time did Jordan discuss audience engagement?", 
        "In frame 5, what did Maya say about color symbolism?",
        "At 15 seconds, what technical analysis did Alex provide?",
        "When did the creative interpreter mention metaphors?",
        "What frame shows Maya discussing emotional impact?",
        "At what point did Jordan talk about viewer connection?",
        "When did Alex analyze the camera work?",
        "What time did Maya interpret the visual symbolism?"
    ]
    
    try:
        rag_interface = RAGQueryInterface()
        processor = TemporalAgentQueryProcessor(rag_interface)
        
        print("💡 Example queries this system can handle:")
        print("-" * 40)
        
        for i, query in enumerate(example_queries, 1):
            print(f"{i:2d}. {query}")
        
        print("\n🔍 Testing query processing...")
        
        # Test a few examples
        test_queries = example_queries[:3]
        
        for query in test_queries:
            print(f"\n{'='*80}")
            result = processor.process_temporal_agent_query(query)
            formatted_response = format_temporal_response(result)
            print(formatted_response)
        
        print("\n✅ Temporal agent query system demonstration complete!")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        print("Make sure you have analyzed videos and indexed them with RAG first.")

if __name__ == "__main__":
    demonstrate_temporal_queries()