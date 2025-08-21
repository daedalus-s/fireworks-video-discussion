#!/usr/bin/env python3
"""
Enhanced RAG Query Interface with Temporal Agent Query Support
Allows complex queries like "At what frame did Maya think blue symbolized peace?"
"""

import os
import sys
import argparse
from typing import Optional, List

from rag_enhanced_vector_system import RAGQueryInterface
from temporal_agent_query_examples import TemporalAgentQueryProcessor, format_temporal_response

def print_results(results: List[dict], query: str):
    """Print formatted search results"""
    if not results:
        print("❌ No results found.")
        print("\n💡 Tips:")
        print("  • Try different search terms")
        print("  • Make sure videos have been analyzed and indexed")
        print("  • Check if RAG indexing was successful")
        print("  • Try temporal queries like 'When did Maya mention blue?'")
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
        
        # Show content
        print(f"   📝 Content: {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
        
        # Show context
        print(f"   🔍 Context: {result['context']}")
        
        # Show related perspectives
        if result['related_perspectives']:
            print(f"   👥 Related insights from: {', '.join([rp['agent_name'] for rp in result['related_perspectives']])}")
            for rp in result['related_perspectives'][:2]:  # Show top 2
                print(f"      • {rp['agent_name']}: {rp['content'][:100]}...")
        
        print("-" * 60)

def is_temporal_agent_query(query: str) -> bool:
    """Detect if query is asking for temporal agent information"""
    temporal_indicators = [
        "at what frame", "when did", "what time", "what frame",
        "in frame", "at frame", "at second", "during",
        "maya think", "alex say", "jordan mention",
        "maya said", "alex thought", "jordan discussed"
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in temporal_indicators)

def interactive_mode():
    """Enhanced interactive query mode with temporal support"""
    print("\n🧠 Enhanced RAG Video Analysis - Interactive Query Mode")
    print("="*70)
    print("Enter natural language queries to search your video library.")
    print("🆕 Now supports temporal agent queries!")
    print("Type 'help' for examples, 'examples' for temporal queries, 'quit' to exit.")
    print("="*70)
    
    try:
        rag_interface = RAGQueryInterface()
        temporal_processor = TemporalAgentQueryProcessor(rag_interface)
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
                show_help_examples()
                continue
                
            if query.lower() in ['examples', 'temporal', 'temp']:
                show_temporal_examples()
                continue
            
            if not query:
                continue
            
            print("⏳ Searching...")
            
            # Check if this is a temporal agent query
            if is_temporal_agent_query(query):
                print("🎯 Detected temporal agent query - using enhanced processor...")
                result = temporal_processor.process_temporal_agent_query(query)
                response = format_temporal_response(result)
                print(response)
            else:
                # Use regular RAG search
                results = rag_interface.query_video_rag(query, num_results=5)
                print_results(results, query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Search failed: {e}")

def show_temporal_examples():
    """Show temporal agent query examples"""
    print("\n🕐 TEMPORAL AGENT QUERY EXAMPLES:")
    print("-" * 50)
    
    print("\n🎯 FRAME-SPECIFIC AGENT QUERIES:")
    print("  • At what frame did Maya think that blue symbolized peace?")
    print("  • When did Alex mention dramatic lighting?")
    print("  • What time did Jordan discuss audience engagement?")
    print("  • In frame 5, what did Maya say about color symbolism?")
    print("  • At 15 seconds, what technical analysis did Alex provide?")
    
    print("\n🎭 AGENT PERSPECTIVE TIMING:")
    print("  • When did the creative interpreter mention metaphors?")
    print("  • What frame shows Maya discussing emotional impact?")
    print("  • At what point did Jordan talk about viewer connection?")
    print("  • When did Alex analyze the camera work?")
    print("  • What time did Maya interpret visual symbolism?")
    
    print("\n⏰ TIME-BASED AGENT INSIGHTS:")
    print("  • At 20 seconds, what did all agents think?")
    print("  • When did Maya and Alex agree on something?")
    print("  • What frame generated the most agent discussion?")
    print("  • At what time did Jordan disagree with Alex?")
    print("  • When did agents discuss production quality?")
    
    print("\n💡 TIP: These queries work best after comprehensive analysis!")

def show_help_examples():
    """Show general help examples"""
    print("\n💡 GENERAL QUERY EXAMPLES:")
    print("-" * 40)
    
    print("\n🎬 TECHNICAL ANALYSIS:")
    print("  • What camera techniques were used?")
    print("  • Describe the lighting in close-up shots")
    print("  • What did Alex say about depth of field?")
    print("  • Find frames with dramatic lighting")
    
    print("\n🎨 CREATIVE INTERPRETATION:")
    print("  • How does color symbolism contribute to the story?")
    print("  • What creative metaphors were discussed?")
    print("  • How did Maya interpret the emotional moments?")
    print("  • What themes were identified in the dialogue?")
    
    print("\n👥 AUDIENCE PERSPECTIVE:")
    print("  • How does this engage the audience?")
    print("  • What did Jordan say about viewer connection?")
    print("  • Which moments are most impactful for viewers?")
    print("  • How accessible is this content?")
    
    print("\n🕐 TEMPORAL QUERIES (NEW!):")
    print("  • At what frame did Maya mention peace?")
    print("  • When did Alex discuss lighting techniques?")
    print("  • What time did Jordan talk about engagement?")
    print("  • In frame 3, what did agents say?")
    print("\n📝 Type 'examples' to see more temporal query examples!")

def main():
    """Enhanced main entry point with temporal query support"""
    parser = argparse.ArgumentParser(
        description="Enhanced RAG Query Interface with Temporal Agent Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python query_video_rag.py
  
  # Regular queries
  python query_video_rag.py "What did Alex say about lighting?"
  
  # NEW: Temporal agent queries
  python query_video_rag.py "At what frame did Maya think blue symbolized peace?"
  python query_video_rag.py "When did Jordan mention audience engagement?"
  python query_video_rag.py "In frame 5, what did Alex say about camera work?"
  
  # Focus on specific agent type
  python query_video_rag.py "camera techniques" --focus technical
  
  # Search specific video
  python query_video_rag.py "emotional moments" --video "video.mp4"

Temporal Query Features:
  ✅ Frame-specific agent insights ("At frame 5, what did Maya say?")
  ✅ Time-based agent queries ("When did Alex mention lighting?")
  ✅ Agent concept timing ("At what point did Jordan discuss engagement?")
  ✅ Cross-agent temporal analysis ("At 15 seconds, what did all agents think?")

Agent Focus Options:
  technical  - Technical Analyst (Alex) perspectives
  creative   - Creative Interpreter (Maya) perspectives  
  audience   - Audience Advocate (Jordan) perspectives
  all        - All agents (default)
        """
    )
    
    parser.add_argument(
        "query", 
        nargs="?", 
        help="Natural language query (omit for interactive mode)"
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
        help="Force temporal agent query processing"
    )
    
    args = parser.parse_args()
    
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
        print(f"🧠 Enhanced RAG Video Analysis Query")
        print("="*60)
        
        rag_interface = RAGQueryInterface()
        
        print(f"🔍 Query: '{args.query}'")
        if args.focus != "all":
            print(f"🎯 Focus: {args.focus} perspectives")
        if args.video:
            print(f"🎬 Video filter: {args.video}")
        
        print("⏳ Searching...")
        
        # Determine query type and process accordingly
        if args.temporal or is_temporal_agent_query(args.query):
            print("🎯 Using temporal agent query processor...")
            temporal_processor = TemporalAgentQueryProcessor(rag_interface)
            result = temporal_processor.process_temporal_agent_query(args.query)
            response = format_temporal_response(result)
            print(response)
        else:
            # Regular RAG search
            results = rag_interface.query_video_rag(
                query=args.query,
                num_results=args.results,
                focus_on=args.focus,
                video_filter=args.video
            )
            print_results(results, args.query)
        
        print(f"\n💡 Try temporal queries like:")
        print(f"   'At what frame did Maya mention [concept]?'")
        print(f"   'When did Alex discuss [technique]?'")
        print(f"   'What time did Jordan talk about [topic]?'")
        
        return 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        print("\nPossible issues:")
        print("  • RAG system not properly initialized")
        print("  • No videos have been indexed yet")
        print("  • Pinecone API key issues")
        print("  • Network connectivity problems")
        return 1

if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
RAG Query Interface for Video Analysis
Allows natural language queries across frame analysis and agent perspectives
"""

import os
import sys
import argparse
from typing import Optional, List

from rag_enhanced_vector_system import RAGQueryInterface

def print_results(results: List[dict], query: str):
    """Print formatted search results"""
    if not results:
        print("❌ No results found.")
        print("\n💡 Tips:")
        print("  • Try different search terms")
        print("  • Make sure videos have been analyzed and indexed")
        print("  • Check if RAG indexing was successful")
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
        
        # Show content
        print(f"   📝 Content: {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
        
        # Show context
        print(f"   🔍 Context: {result['context']}")
        
        # Show related perspectives
        if result['related_perspectives']:
            print(f"   👥 Related insights from: {', '.join([rp['agent_name'] for rp in result['related_perspectives']])}")
            for rp in result['related_perspectives'][:2]:  # Show top 2
                print(f"      • {rp['agent_name']}: {rp['content'][:100]}...")
        
        print("-" * 60)

def interactive_mode():
    """Interactive query mode"""
    print("\n🧠 RAG Video Analysis - Interactive Query Mode")
    print("="*60)
    print("Enter natural language queries to search your video library.")
    print("Type 'help' for examples, 'quit' to exit.")
    print("="*60)
    
    try:
        rag_interface = RAGQueryInterface()
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
                show_help_examples()
                continue
            
            if not query:
                continue
            
            print("⏳ Searching...")
            results = rag_interface.query_video_rag(query, num_results=5)
            print_results(results, query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Search failed: {e}")

def show_help_examples():
    """Show example queries"""
    print("\n💡 EXAMPLE QUERIES:")
    print("-" * 40)
    
    print("\n🎬 TECHNICAL ANALYSIS:")
    print("  • What camera techniques were used?")
    print("  • Describe the lighting in close-up shots")
    print("  • What did Alex say about depth of field?")
    print("  • Find frames with dramatic lighting")
    
    print("\n🎨 CREATIVE INTERPRETATION:")
    print("  • How does color symbolism contribute to the story?")
    print("  • What creative metaphors were discussed?")
    print("  • How did Maya interpret the emotional moments?")
    print("  • What themes were identified in the dialogue?")
    
    print("\n👥 AUDIENCE PERSPECTIVE:")
    print("  • How does this engage the audience?")
    print("  • What did Jordan say about viewer connection?")
    print("  • Which moments are most impactful for viewers?")
    print("  • How accessible is this content?")
    
    print("\n⏰ TEMPORAL QUERIES:")
    print("  • What happens around 15 seconds?")
    print("  • Describe the opening sequence")
    print("  • What agent insights were shared about frame 3?")
    print("  • Find perspectives about the middle section")
    
    print("\n🔍 CROSS-PERSPECTIVE:")
    print("  • How do all agents view the character development?")
    print("  • What consensus exists about the production quality?")
    print("  • Compare technical vs creative perspectives")
    print("  • What disagreements exist between agents?")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Query your RAG-indexed video analysis library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python query_video_rag.py
  
  # Single query
  python query_video_rag.py "What did Alex say about lighting?"
  
  # Focus on specific agent type
  python query_video_rag.py "camera techniques" --focus technical
  
  # Search specific video
  python query_video_rag.py "emotional moments" --video "video.mp4"
  
  # Get more results
  python query_video_rag.py "dialogue analysis" --results 10

Agent Focus Options:
  technical  - Technical Analyst (Alex) perspectives
  creative   - Creative Interpreter (Maya) perspectives  
  audience   - Audience Advocate (Jordan) perspectives
  all        - All agents (default)
        """
    )
    
    parser.add_argument(
        "query", 
        nargs="?", 
        help="Natural language query (omit for interactive mode)"
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
    
    args = parser.parse_args()
    
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
        print(f"🧠 RAG Video Analysis Query")
        print("="*50)
        
        rag_interface = RAGQueryInterface()
        
        print(f"🔍 Query: '{args.query}'")
        if args.focus != "all":
            print(f"🎯 Focus: {args.focus} perspectives")
        if args.video:
            print(f"🎬 Video filter: {args.video}")
        
        print("⏳ Searching...")
        
        results = rag_interface.query_video_rag(
            query=args.query,
            num_results=args.results,
            focus_on=args.focus,
            video_filter=args.video
        )
        
        print_results(results, args.query)
        
        if results:
            print(f"\n💡 Found {len(results)} results. Try refining your query for more specific results.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        print("\nPossible issues:")
        print("  • RAG system not properly initialized")
        print("  • No videos have been indexed yet")
        print("  • Pinecone API key issues")
        print("  • Network connectivity problems")
        return 1

if __name__ == "__main__":
    sys.exit(main())