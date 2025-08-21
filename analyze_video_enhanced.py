#!/usr/bin/env python3
"""
Enhanced Command Line Interface for Descriptive Video Analysis
Usage: python analyze_video_enhanced.py [options] video_file
"""

import os
import asyncio
import argparse
import sys
from pathlib import Path

# Import the enhanced analysis system
from enhanced_descriptive_analysis import analyze_video_highly_descriptive, DescriptiveAnalysisPipeline
from vector_search_system import VideoSearchInterface

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    # Check for required packages
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        from pinecone import Pinecone
    except ImportError:
        missing_deps.append("pinecone")
    
    # Check for API keys
    if not os.getenv("FIREWORKS_API_KEY"):
        print("⚠️ Warning: FIREWORKS_API_KEY not found in environment")
    
    if not os.getenv("PINECONE_API_KEY"):
        print("⚠️ Warning: PINECONE_API_KEY not found - vector search will be disabled")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def search_videos(query: str, video_path: str = None, num_results: int = 5):
    """Search for events in indexed videos"""
    try:
        print(f"\n🔍 Searching for: '{query}'")
        if video_path:
            print(f"📁 In video: {video_path}")
        
        search_interface = VideoSearchInterface()
        
        # Search for frames
        results = search_interface.find_event(
            event=query,
            video=video_path,
            num_results=num_results
        )
        
        # Search for timeframes
        timeframes = search_interface.find_timeframes(
            event=query,
            video=video_path,
            duration=10.0,
            num_results=min(3, num_results)
        )
        
        print(f"\n📊 SEARCH RESULTS")
        print("-"*50)
        
        if results:
            print(f"Found {len(results)} matching frames:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Frame {result['frame_number']} at {result['timestamp']:.1f}s")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Description: {result['description'][:150]}...")
        else:
            print("No matching frames found.")
            print("Make sure you've analyzed some videos first with enhanced descriptions.")
        
        if timeframes:
            print(f"\n🎬 Suggested viewing times:")
            for i, tf in enumerate(timeframes, 1):
                print(f"{i}. Watch from {tf['start_time']:.1f}s to {tf['end_time']:.1f}s")
                print(f"   (Best moment at {tf['center_timestamp']:.1f}s)")
                print(f"   Confidence: {tf['similarity_score']:.3f}")
        
        return len(results)
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        print("Make sure PINECONE_API_KEY is set and you've analyzed some videos.")
        return 0

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Descriptive Video Analysis with Vector Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_video_enhanced.py video.mp4
  
  # Comprehensive analysis with subtitles
  python analyze_video_enhanced.py video.mp4 -s subtitles.srt --depth comprehensive
  
  # Detailed analysis with more frames
  python analyze_video_enhanced.py video.mp4 --frames 20 --depth detailed
  
  # Search for events (after analyzing videos)
  python analyze_video_enhanced.py --search "person walking"
  
  # Search in specific video
  python analyze_video_enhanced.py video.mp4 --search "car driving"

Analysis Depth Options:
  basic        - Standard frame analysis (faster, lower cost)
  detailed     - Enhanced descriptions with more detail
  comprehensive - Maximum detail with full cinematography analysis (slower, higher cost)
        """
    )
    
    # Video file argument (optional for search mode)
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to video file"
    )
    
    # Subtitle options
    parser.add_argument(
        "--subtitles", "-s",
        help="Path to subtitle file (VTT or SRT)",
        default=None
    )
    
    # Analysis parameters
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=10,
        help="Maximum frames to analyze (default: 10)"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        default=0.2,
        help="Frame extraction rate in fps (default: 0.2)"
    )
    
    parser.add_argument(
        "--depth", "-d",
        choices=["basic", "detailed", "comprehensive"],
        default="comprehensive",
        help="Analysis depth (default: comprehensive)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        default="enhanced_analysis_output",
        help="Output directory (default: enhanced_analysis_output)"
    )
    
    # Vector search options
    parser.add_argument(
        "--no-vector-search",
        action="store_true",
        help="Disable vector search functionality"
    )
    
    parser.add_argument(
        "--search",
        help="Search for specific events in analyzed videos"
    )
    
    parser.add_argument(
        "--search-results", "-n",
        type=int,
        default=5,
        help="Number of search results to return (default: 5)"
    )
    
    # Cost and performance options
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate analysis cost before running"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster settings (basic depth, fewer frames)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Handle search mode
    if args.search:
        print("🔍 ENHANCED SEARCH MODE")
        print("="*60)
        results_found = search_videos(
            query=args.search,
            video_path=args.video if args.video and os.path.exists(args.video) else None,
            num_results=args.search_results
        )
        
        if results_found == 0:
            print("\n💡 Tips for better search results:")
            print("  • Analyze videos with --depth comprehensive for best search quality")
            print("  • Try different search terms or phrases")
            print("  • Make sure you've analyzed videos first")
        
        return 0
    
    # Validate video file for analysis mode
    if not args.video:
        print("❌ Error: Video file required for analysis mode")
        print("Use --search 'query' for search mode, or provide a video file for analysis")
        return 1
    
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return 1
    
    # Check subtitle file
    if args.subtitles and not os.path.exists(args.subtitles):
        print(f"⚠️ Warning: Subtitle file not found: {args.subtitles}")
        print("Continuing without subtitles...")
        args.subtitles = None
    
    # Apply fast mode settings
    if args.fast:
        args.depth = "basic"
        args.frames = min(args.frames, 5)
        print("🚀 Fast mode: Using basic analysis with maximum 5 frames")
    
    # Estimate cost
    if args.estimate_cost:
        print("💰 COST ESTIMATION")
        print("-"*30)
        
        # Rough cost estimates based on analysis depth
        cost_per_frame = {
            "basic": 0.0003,
            "detailed": 0.0005,
            "comprehensive": 0.0008
        }
        
        estimated_cost = args.frames * cost_per_frame[args.depth]
        if args.subtitles:
            estimated_cost += 0.002  # Subtitle analysis
        estimated_cost += 0.001  # Overall analysis
        
        print(f"Estimated cost for {args.depth} analysis of {args.frames} frames: ${estimated_cost:.4f}")
        print(f"Analysis depth: {args.depth}")
        
        response = input("Continue with analysis? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Analysis cancelled.")
            return 0
    
    # Configure vector search
    enable_vector_search = not args.no_vector_search
    if enable_vector_search and not os.getenv("PINECONE_API_KEY"):
        print("⚠️ PINECONE_API_KEY not found - disabling vector search")
        enable_vector_search = False
    
    # Show configuration
    print("\n" + "="*70)
    print("🎬 ENHANCED DESCRIPTIVE VIDEO ANALYSIS")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Subtitles: {args.subtitles or 'None'}")
    print(f"Analysis depth: {args.depth.upper()}")
    print(f"Frames to analyze: {args.frames}")
    print(f"Frame extraction rate: {args.fps} fps")
    print(f"Vector search: {'Enabled' if enable_vector_search else 'Disabled'}")
    print(f"Output directory: {args.output}")
    print("="*70)
    
    # Run enhanced analysis
    try:
        start_msg = {
            "basic": "Running basic enhanced analysis...",
            "detailed": "Running detailed descriptive analysis...",
            "comprehensive": "Running comprehensive cinematographic analysis..."
        }
        
        print(f"\n{start_msg[args.depth]}")
        
        results = await analyze_video_highly_descriptive(
            video_path=args.video,
            subtitle_path=args.subtitles,
            max_frames=args.frames,
            fps_extract=args.fps,
            analysis_depth=args.depth,
            enable_vector_search=enable_vector_search,
            output_dir=args.output
        )
        
        # Show completion summary
        print(f"\n🎉 Enhanced analysis complete!")
        print(f"📁 Results saved to: {args.output}")
        print(f"💰 Total cost: ${results['total_cost']:.4f}")
        
        if enable_vector_search:
            print(f"\n🔍 Your video is now searchable! Try:")
            print(f"  python analyze_video_enhanced.py --search 'person walking'")
            print(f"  python analyze_video_enhanced.py --search 'indoor scene'")
            print(f"  python analyze_video_enhanced.py --search 'close-up shot'")
            
            # Show enhanced search capabilities
            if args.depth == "comprehensive":
                print(f"\n✨ Comprehensive analysis enables detailed searches:")
                print(f"  - Cinematography: 'close-up shot', 'wide angle', 'dramatic lighting'")
                print(f"  - Technical: 'shallow depth of field', 'natural lighting', 'handheld camera'")
                print(f"  - Narrative: 'emotional conversation', 'tense moment', 'comic relief'")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)