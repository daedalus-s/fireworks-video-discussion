#!/usr/bin/env python3
"""
Complete RAG Query Interface with Universal Temporal Processing
Works with any video content and configurable agents
"""

import os
import sys
import argparse
from typing import Optional, List
import re

from rag_enhanced_vector_system import RAGQueryInterface
from universal_temporal_processor import UniversalTemporalAgentQueryProcessor, format_universal_temporal_response

def print_results(results: List[dict], query: str, show_full_content: bool = False):
    """Print formatted search results with option for full content"""
    if not results:
        print("❌ No results found.")
        print("\n💡 Tips:")
        print("  • Try different search terms")
        print("  • Make sure videos have been analyzed and indexed")
        print("  • Check if RAG indexing was successful")
        print("  • Try temporal queries like 'When did Maya mention [topic]?'")
        return
    
    print(f"\n📊 Found {len(results)} results for: '{query}'")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['document_type'].upper()}] Confidence: {result['confidence']:.3f}")
        
        # Show agent info if available
        if result['agent']:
            print(f"   🤖 Agent: {result['agent']['name']} ({result['agent']['role']})")
        
        # Show temporal info
        if result['timestamp']:
            print(f"   ⏰ Time: {result['timestamp']:.1f}s", end="")
            if result['frame_number']:
                print(f" (Frame {result['frame_number']})")
            else:
                print()
        
        # Show video
        print(f"   🎬 Video: {result['video']}")
        
        # Show content - FIXED: Show full content or more characters
        content = result['content']
        if show_full_content or len(content) <= 500:
            print(f"   📝 Content: {content}")
        else:
            # Show more characters but still truncate very long content
            print(f"   📝 Content: {content[:800]}{'...' if len(content) > 800 else ''}")
        
        # Show context
        print(f"   🔍 Context: {result['context']}")
        
        # Show related perspectives
        if result['related_perspectives']:
            print(f"   👥 Related insights from: {', '.join([rp['agent_name'] for rp in result['related_perspectives']])}")
            for rp in result['related_perspectives'][:2]:  # Show top 2
                rp_content = rp['content']
                if show_full_content:
                    print(f"      • {rp['agent_name']}: {rp_content}")
                else:
                    print(f"      • {rp['agent_name']}: {rp_content[:200]}...")
        
        print("-" * 60)

def is_temporal_query(query: str) -> bool:
    """Detect if query benefits from temporal processing - universal patterns"""
    temporal_indicators = [
        # Frame-specific queries
        r'(?:at\s+)?(?:what\s+)?frame\s+\d+',
        r'(?:in\s+)?frame\s+\d+',
        r'frame\s+number\s+\d+',
        
        # Time-specific queries
        r'(?:at\s+)?\d+(?:\.\d+)?\s*(?:seconds?|s)\b',
        r'(?:at\s+)?\d+:\d+',
        r'(?:around\s+)?\d+(?:\.\d+)?\s*(?:sec|second)s?\s+(?:mark|point)',
        
        # When queries
        r'when\s+did\s+.+?\s+(?:happen|occur|start|begin|end|appear)',
        r'when\s+(?:does|did)\s+.+?\s+(?:move|change|show|display)',
        r'what\s+time\s+(?:does|did)\s+.+?\s+(?:happen|occur|start)',
        
        # Agent temporal queries
        r'(?:when|at\s+what\s+(?:frame|time))\s+did\s+(?:alex|maya|jordan|\w+)',
        r'(?:what\s+did|how\s+did)\s+(?:alex|maya|jordan|\w+)\s+(?:say|mention|think)',
        r'(?:in\s+frame\s+\d+|at\s+\d+(?:\.\d+)?\s*s),?\s+what\s+did\s+(?:alex|maya|jordan|\w+)',
        
        # Content temporal queries
        r'(?:at\s+what\s+(?:frame|time))\s+(?:does|did|is|was)\s+.+?\s+(?:visible|shown|happening)',
        r'(?:which\s+frame|what\s+time)\s+.+?\s+(?:appears|shows|displays)',
        r'(?:describe|show|explain)\s+(?:what\s+happens?\s+)?(?:in\s+frame\s+\d+|at\s+\d+(?:\.\d+)?\s*s)'
    ]
    
    query_lower = query.lower()
    return any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in temporal_indicators)

def interactive_mode():
    """Universal interactive query mode"""
    print("\n🧠 Universal RAG Video Analysis - Interactive Query Mode")
    print("="*70)
    print("Enter natural language queries to search your video library.")
    print("🌟 Works with any video content - ask about anything you see!")
    print("Type 'help' for examples, 'temporal' for advanced queries, 'quit' to exit.")
    print("="*70)
    
    try:
        rag_interface = RAGQueryInterface()
        universal_processor = UniversalTemporalAgentQueryProcessor(rag_interface)
    except Exception as e:
        print(f"❌ Failed to initialize RAG interface: {e}")
        print("Make sure PINECONE_API_KEY is set and videos have been indexed.")
        return
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                show_universal_help_examples()
                continue
                
            if query.lower() in ['temporal', 'advanced', 'temp']:
                show_temporal_examples()
                continue
            
            if not query:
                continue
            
            print("⏳ Searching...")
            
            # Check if this benefits from temporal processing
            if is_temporal_query(query):
                print("🎯 Using enhanced temporal processor...")
                result = universal_processor.process_temporal_query(query)
                response = format_universal_temporal_response(result)
                print(response)
            else:
                # Use regular RAG search
                results = rag_interface.query_video_rag(query, num_results=5)
                print_results(results, query, show_full_content=True)  # Always show full in interactive
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Search failed: {e}")

def show_universal_help_examples():
    """Show universal help examples for any video content"""
    print("\n💡 UNIVERSAL QUERY EXAMPLES:")
    print("-" * 40)
    
    print("\n🎬 GENERAL VIDEO CONTENT:")
    print("  • What happens in frame 5?")
    print("  • When does the person appear?")
    print("  • Describe what's visible at 30 seconds")
    print("  • What did Alex say about the lighting?")
    print("  • When does the action start?")
    print("  • What objects are visible in frame 10?")
    
    print("\n🤖 AGENT PERSPECTIVES:")
    print("  • What did Maya think about the colors?")
    print("  • How did Jordan analyze the audience appeal?")
    print("  • When did Alex discuss camera techniques?")
    print("  • What creative insights did Maya provide?")
    print("  • What technical aspects did Alex mention?")
    print("  • How did Jordan view the engagement level?")
    
    print("\n⏰ TEMPORAL QUERIES:")
    print("  • At what frame does movement occur?")
    print("  • When did the scene change?")
    print("  • What happens around 1:30?")
    print("  • Describe frame 8 content")
    print("  • At 45 seconds, what is visible?")
    print("  • When does the character speak?")
    
    print("\n🔍 CONTENT ANALYSIS:")
    print("  • Find scenes with dramatic lighting")
    print("  • When are close-up shots used?")
    print("  • What emotions are expressed?")
    print("  • Locate outdoor scenes")
    print("  • Find moments with multiple people")
    print("  • When is music most prominent?")

def show_temporal_examples():
    """Show advanced temporal query examples"""
    print("\n⏰ ADVANCED TEMPORAL QUERY EXAMPLES:")
    print("-" * 50)
    
    print("\n🎯 FRAME-SPECIFIC QUERIES:")
    print("  • At what frame did [any event] happen?")
    print("  • What did Maya say about frame 3?")
    print("  • Describe what's in frame 7")
    print("  • In frame 12, what analysis did Alex provide?")
    print("  • When did Jordan mention frame 5?")
    
    print("\n⏱️ TIME-BASED QUERIES:")
    print("  • What happens at 2:15?")
    print("  • At 30 seconds, what did agents discuss?")
    print("  • When did the lighting change?")
    print("  • What occurs around the 1-minute mark?")
    print("  • At what time does [any event] start?")
    
    print("\n🤖 AGENT TEMPORAL ANALYSIS:")
    print("  • When did Alex analyze [any topic]?")
    print("  • What time did Maya interpret [any element]?")
    print("  • At what frame did Jordan discuss [any aspect]?")
    print("  • When did agents agree on [any topic]?")
    print("  • What did [agent] say about [timestamp]?")
    
    print("\n🔄 CROSS-REFERENCE QUERIES:")
    print("  • Compare what agents said about frame 4")
    print("  • When did multiple agents discuss the same topic?")
    print("  • What consensus exists about [any element]?")
    print("  • How do agent perspectives differ at [time]?")
    
    print("\n💡 TIP: Replace [bracketed items] with specific content from your video!")

def show_configurable_agent_examples():
    """Show examples for configurable agents"""
    print("\n🤖 CONFIGURABLE AGENT QUERY EXAMPLES:")
    print("-" * 50)
    
    print("\n🎬 FILM ANALYSIS AGENTS:")
    print("  • What did the Cinematographer say about camera work?")
    print("  • How did the Film Critic rate the artistic merit?")
    print("  • When did the Sound Designer discuss audio elements?")
    print("  • At what frame did the Cinematographer mention composition?")
    
    print("\n🎓 EDUCATIONAL CONTENT AGENTS:")
    print("  • What did the Learning Specialist say about engagement?")
    print("  • How did the Subject Expert verify accuracy?")
    print("  • When did the Engagement Analyst discuss retention?")
    print("  • At what time did agents discuss learning outcomes?")
    
    print("\n📈 MARKETING CONTENT AGENTS:")
    print("  • What did the Brand Strategist say about messaging?")
    print("  • How did the Conversion Specialist rate CTAs?")
    print("  • When did the Creative Director discuss visual appeal?")
    print("  • At what frame did agents discuss brand impact?")
    
    print("\n🔧 CUSTOM AGENT QUERIES:")
    print("  • What did [YourCustomAgent] say about [topic]?")
    print("  • When did the [CustomRole] mention [concept]?")
    print("  • How did [AgentName] analyze [element]?")
    print("  • At what time did [AgentRole] discuss [aspect]?")

def main():
    """Universal main entry point - works with any video content and configurable agents"""
    parser = argparse.ArgumentParser(
        description="Universal RAG Query Interface - Works with Any Video Content and Configurable Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
UNIVERSAL EXAMPLES (works with any video and any configured agents):
  # Interactive mode
  python query_video_rag.py
  
  # General content queries
  python query_video_rag.py "What happens in frame 5?"
  python query_video_rag.py "When does the person move?"
  python query_video_rag.py "Describe what's at 30 seconds"
  
  # Agent-specific queries (works with default or custom agents)
  python query_video_rag.py "What did Alex say about lighting?" --focus technical
  python query_video_rag.py "How did Maya interpret the colors?" --focus creative
  python query_video_rag.py "What did Jordan think about engagement?" --focus audience
  
  # Configurable agent queries
  python query_video_rag.py "What did the Cinematographer say about camera work?"
  python query_video_rag.py "How did the Brand Strategist rate the messaging?"
  python query_video_rag.py "When did the Learning Specialist discuss engagement?"
  
  # Temporal queries (any content)
  python query_video_rag.py "At what frame does action occur?"
  python query_video_rag.py "When did Maya mention emotions?"
  python query_video_rag.py "What happens around 1:30?"

AGENT FOCUS OPTIONS:
  technical  - Technical analysis perspectives
  creative   - Creative interpretation perspectives  
  audience   - Audience/user experience perspectives
  all        - All configured agents (default)

SUPPORTED QUERY TYPES:
  ✅ Frame-specific: "What's in frame X?"
  ✅ Time-based: "What happens at X seconds?"
  ✅ Agent queries: "What did [agent] say about X?"
  ✅ Content search: "Find scenes with X"
  ✅ When questions: "When does X happen?"
  ✅ Temporal agent: "At what time did [agent] discuss X?"
  ✅ Configurable agents: Works with any custom agents you create
        """
    )
    
    parser.add_argument(
        "query", 
        nargs="?", 
        help="Natural language query about your video content"
    )
    
    parser.add_argument(
        "--results", "-n",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--focus", "-f",
        choices=["all", "technical", "creative", "audience"],
        default="all",
        help="Focus on specific agent perspectives (default: all)"
    )
    
    parser.add_argument(
        "--video", "-v",
        help="Filter results to specific video"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Force interactive mode"
    )
    
    parser.add_argument(
        "--temporal", "-t",
        action="store_true",
        help="Force temporal query processing"
    )
    
    parser.add_argument(
        "--examples", "-e",
        choices=["general", "temporal", "configurable"],
        help="Show specific example types and exit"
    )

    parser.add_argument(
    "--full",
    action="store_true",
    help="Show full content in results (no truncation)"
)
    
    args = parser.parse_args()
    
    # Handle examples
    if args.examples:
        if args.examples == "general":
            show_universal_help_examples()
        elif args.examples == "temporal":
            show_temporal_examples()
        elif args.examples == "configurable":
            show_configurable_agent_examples()
        return 0
    
    # Check dependencies
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY not found in environment")
        print("Set your Pinecone API key: export PINECONE_API_KEY=your-key")
        return 1
    
    # Interactive mode
    if not args.query or args.interactive:
        interactive_mode()
        return 0
    
    # Single query mode
    try:
        print(f"🧠 Universal RAG Video Analysis Query")
        print("="*60)
        
        rag_interface = RAGQueryInterface()
        
        print(f"🔍 Query: '{args.query}'")
        if args.focus != "all":
            print(f"🎯 Focus: {args.focus} perspectives")
        if args.video:
            print(f"🎬 Video filter: {args.video}")
        
        print("⏳ Searching...")
        
        # Determine query type and process accordingly
        if args.temporal or is_temporal_query(args.query):
            print("🎯 Using universal temporal processor...")
            universal_processor = UniversalTemporalAgentQueryProcessor(rag_interface)
            result = universal_processor.process_temporal_query(args.query)
            response = format_universal_temporal_response(result)
            print(response)
        else:
            # Regular RAG search
            results = rag_interface.query_video_rag(
                query=args.query,
                num_results=args.results,
                focus_on=args.focus,
                video_filter=args.video
            )
            print_results(results, args.query, show_full_content=args.full)
        
        print(f"\n💡 Try these universal query types:")
        print(f"   'What happens in frame [number]?'")
        print(f"   'When did [agent] mention [topic]?'")
        print(f"   'At [time], what is visible?'")
        print(f"   'Describe what occurs at [timestamp]'")
        print(f"\n📖 For more examples: python query_video_rag.py --examples [general|temporal|configurable]")
        
        return 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        print("\nPossible issues:")
        print("  • RAG system not properly initialized")
        print("  • No videos have been indexed yet")
        print("  • Pinecone API key issues")
        print("  • Network connectivity problems")
        print("\n💡 Try running analysis first:")
        print("  python integrated_configurable_pipeline.py your_video.mp4")
        return 1

if __name__ == "__main__":
    sys.exit(main())