"""
Enhanced Video Analysis Pipeline with Vector Search Capability
Integrates vector search with existing video analysis and discussion system
"""

import os
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
from typing import Optional, List, Dict, Any

from video_analysis_system import VideoAnalysisSystem, VideoAnalysisResult
from multi_agent_discussion import MultiAgentDiscussion
from vector_search_system import VideoSearchInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedVideoAnalysisPipeline:
    """Enhanced pipeline with vector search capabilities"""
    
    def __init__(self, enable_vector_search: bool = True):
        """
        Initialize enhanced pipeline
        
        Args:
            enable_vector_search: Whether to enable vector search functionality
        """
        self.enable_vector_search = enable_vector_search
        
        # Initialize core systems
        self.analysis_system = VideoAnalysisSystem()
        self.discussion_system = MultiAgentDiscussion()
        
        # Initialize vector search if enabled
        self.search_interface = None
        if enable_vector_search:
            try:
                self.search_interface = VideoSearchInterface()
                logger.info("✅ Vector search enabled")
            except Exception as e:
                logger.warning(f"⚠️ Vector search disabled: {e}")
                self.enable_vector_search = False
    
    async def analyze_and_index_video(self,
                                    video_path: str,
                                    subtitle_path: Optional[str] = None,
                                    max_frames: int = 10,
                                    fps_extract: float = 0.2,
                                    discussion_rounds: int = 3,
                                    output_dir: str = "analysis_output") -> Dict[str, Any]:
        """
        Complete pipeline: analyze video, index for search, and generate discussion
        
        Args:
            video_path: Path to video file
            subtitle_path: Path to subtitle file (optional)
            max_frames: Maximum frames to analyze
            fps_extract: Frames per second to extract
            discussion_rounds: Number of discussion rounds
            output_dir: Directory for output files
            
        Returns:
            Complete analysis results with search capabilities
        """
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamp for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*70)
        print("🎬 ENHANCED VIDEO ANALYSIS PIPELINE WITH VECTOR SEARCH")
        print("="*70)
        print(f"Video: {video_path}")
        print(f"Subtitles: {subtitle_path if subtitle_path else 'None'}")
        print(f"Max frames: {max_frames}")
        print(f"Frame extraction rate: {fps_extract} fps")
        print(f"Discussion rounds: {discussion_rounds}")
        print(f"Vector search: {'Enabled' if self.enable_vector_search else 'Disabled'}")
        print(f"Output directory: {output_dir}")
        print("="*70)
        
        # Step 1: Analyze the video
        print("\n📊 PHASE 1: VIDEO ANALYSIS")
        print("-"*40)
        
        try:
            video_results = await self.analysis_system.analyze_video(
                video_path=video_path,
                subtitle_path=subtitle_path,
                max_frames=max_frames,
                fps_extract=fps_extract
            )
            
            # Save video analysis
            analysis_file = output_path / f"video_analysis_{timestamp}.json"
            self.analysis_system.save_results(video_results, str(analysis_file))
            
            # Print summary
            self.analysis_system.print_summary(video_results)
            
            print(f"\n✅ Video analysis saved to: {analysis_file}")
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            raise
        
        # Step 2: Index for vector search
        search_status = "Not enabled"
        if self.enable_vector_search and self.search_interface:
            print("\n🔍 PHASE 2: VECTOR SEARCH INDEXING")
            print("-"*40)
            
            try:
                # Convert VideoAnalysisResult to dict for indexing
                video_data = {
                    "video_path": video_results.video_path,
                    "frame_count": video_results.frame_count,
                    "subtitle_count": video_results.subtitle_count,
                    "frame_analyses": video_results.frame_analyses,
                    "subtitle_analyses": video_results.subtitle_analyses,
                    "overall_analysis": video_results.overall_analysis
                }
                
                search_status = self.search_interface.add_video_to_search(video_data)
                print(search_status)
                
            except Exception as e:
                logger.error(f"Vector indexing failed: {e}")
                search_status = f"Failed: {e}"
        
        # Step 3: Generate multi-agent discussion
        print("\n💬 PHASE 3: MULTI-AGENT DISCUSSION")
        print("-"*40)
        
        try:
            # Convert results to dict for discussion
            video_data = {
                "video_path": video_results.video_path,
                "frame_count": video_results.frame_count,
                "subtitle_count": video_results.subtitle_count,
                "frame_analyses": video_results.frame_analyses,
                "subtitle_analyses": video_results.subtitle_analyses,
                "overall_analysis": video_results.overall_analysis
            }
            
            # Generate discussion
            discussion = await self.discussion_system.discuss_video(
                video_analysis=video_data,
                num_rounds=discussion_rounds
            )
            
            # Print discussion
            self.discussion_system.print_discussion()
            
            # Save discussion
            discussion_file = output_path / f"discussion_{timestamp}.json"
            self.discussion_system.save_discussion(str(discussion_file))
            
            print(f"\n✅ Discussion saved to: {discussion_file}")
            
        except Exception as e:
            logger.error(f"Discussion generation failed: {e}")
            raise
        
        # Step 4: Generate enhanced report with search capabilities
        print("\n📄 PHASE 4: GENERATING ENHANCED REPORT")
        print("-"*40)
        
        try:
            # Get search statistics if available
            search_stats = {}
            if self.enable_vector_search and self.search_interface:
                search_stats = self.search_interface.search_stats()
            
            report = {
                "timestamp": timestamp,
                "video_path": video_path,
                "subtitle_path": subtitle_path,
                "analysis_summary": {
                    "frames_analyzed": video_results.frame_count,
                    "subtitles_processed": video_results.subtitle_count,
                    "processing_time": video_results.processing_time,
                    "analysis_cost": video_results.total_cost
                },
                "vector_search": {
                    "enabled": self.enable_vector_search,
                    "indexing_status": search_status,
                    "statistics": search_stats
                },
                "key_insights": {
                    "visual_highlights": [
                        f"Frame {fa['timestamp']:.1f}s: {fa['analysis'][:150]}..."
                        for fa in video_results.frame_analyses[:3]
                        if fa['analysis'] and not 'failed' in fa['analysis'].lower()
                    ],
                    "dialogue_highlights": [
                        f"{sa['subtitle_range']}: {sa['analysis'][:150]}..."
                        for sa in video_results.subtitle_analyses[:2]
                        if sa['analysis'] and not 'failed' in sa['analysis'].lower()
                    ] if video_results.subtitle_analyses else [],
                    "overall_assessment": video_results.overall_analysis
                },
                "discussion_summary": {
                    "total_turns": len(discussion),
                    "agents_participated": list(set(turn.agent_name for turn in discussion)),
                    "key_points": [
                        f"{turn.agent_name}: {turn.content[:200]}..."
                        for turn in discussion[:6]  # First 6 turns
                    ]
                },
                "search_capabilities": {
                    "available": self.enable_vector_search,
                    "example_queries": [
                        "person walking",
                        "car driving",
                        "people talking",
                        "text on screen",
                        "indoor scene",
                        "outdoor scene"
                    ] if self.enable_vector_search else []
                },
                "total_cost": self.discussion_system.client.get_usage_summary()['total_cost_usd']
            }
            
            # Save enhanced report
            report_file = output_path / f"enhanced_report_{timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            # Create enhanced text report with search instructions
            text_report_file = output_path / f"enhanced_report_{timestamp}.txt"
            with open(text_report_file, 'w', encoding='utf-8') as f:
                f.write("ENHANCED VIDEO ANALYSIS REPORT WITH VECTOR SEARCH\n")
                f.write("="*70 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Video: {video_path}\n")
                f.write(f"Subtitles: {subtitle_path if subtitle_path else 'None'}\n\n")
                
                f.write("ANALYSIS SUMMARY\n")
                f.write("-"*40 + "\n")
                f.write(f"Frames analyzed: {video_results.frame_count}\n")
                f.write(f"Processing time: {video_results.processing_time:.2f} seconds\n")
                f.write(f"Overall assessment:\n{video_results.overall_analysis}\n\n")
                
                f.write("VECTOR SEARCH CAPABILITIES\n")
                f.write("-"*40 + "\n")
                if self.enable_vector_search:
                    f.write("✅ Vector search is ENABLED for this video\n")
                    f.write(f"Indexing status: {search_status}\n")
                    if search_stats:
                        f.write(f"Total indexed frames: {search_stats.get('total_vectors', 'Unknown')}\n")
                    f.write("\nExample search queries you can use:\n")
                    for query in report["search_capabilities"]["example_queries"]:
                        f.write(f"  - \"{query}\"\n")
                    f.write("\nUse the search_video_events() function to find specific events!\n")
                else:
                    f.write("❌ Vector search is DISABLED\n")
                    f.write("To enable: Set PINECONE_API_KEY and install: pip install pinecone-client sentence-transformers\n")
                
                f.write("\nMULTI-AGENT DISCUSSION\n")
                f.write("-"*40 + "\n")
                for turn in discussion:
                    f.write(f"\n{turn.agent_name} ({turn.agent_role}):\n")
                    f.write(f"{turn.content}\n")
                    f.write("-"*30 + "\n")
                
                f.write(f"\nTotal cost: ${report['total_cost']:.4f}\n")
            
            print(f"✅ Enhanced report saved to: {report_file}")
            print(f"✅ Text report saved to: {text_report_file}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
        
        print("\n" + "="*70)
        print("✅ ENHANCED PIPELINE FINISHED SUCCESSFULLY!")
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"💰 Total cost: ${report['total_cost']:.4f}")
        if self.enable_vector_search:
            print("🔍 Vector search capabilities added - you can now search for specific events!")
        print("="*70)
        
        return report

def search_video_events(event_description: str, 
                       video_path: Optional[str] = None,
                       num_results: int = 5,
                       timeframe_duration: float = 10.0) -> Dict[str, Any]:
    """
    Search for specific events in indexed videos
    
    Args:
        event_description: Natural language description of what to find
        video_path: Optional specific video to search in  
        num_results: Number of results to return
        timeframe_duration: Length of timeframes in seconds
        
    Returns:
        Dictionary with frame results and timeframes
    """
    try:
        # Initialize search interface
        search_interface = VideoSearchInterface()
        
        print(f"\n🔍 Searching for: '{event_description}'")
        if video_path:
            print(f"📁 In video: {video_path}")
        
        # Search for individual frames
        frame_results = search_interface.find_event(
            event=event_description,
            video=video_path,
            num_results=num_results
        )
        
        # Search for timeframes
        timeframe_results = search_interface.find_timeframes(
            event=event_description,
            video=video_path,
            duration=timeframe_duration,
            num_results=min(3, num_results)
        )
        
        # Format results
        print(f"\n📊 SEARCH RESULTS")
        print("-"*40)
        
        if frame_results:
            print(f"Found {len(frame_results)} matching frames:")
            for i, result in enumerate(frame_results, 1):
                print(f"{i}. Frame {result['frame_number']} at {result['timestamp']:.1f}s")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Description: {result['description'][:100]}...")
                print()
        else:
            print("No matching frames found.")
        
        if timeframe_results:
            print(f"Suggested timeframes to watch:")
            for i, tf in enumerate(timeframe_results, 1):
                print(f"{i}. {tf['start_time']:.1f}s - {tf['end_time']:.1f}s")
                print(f"   Confidence: {tf['similarity_score']:.3f}")
                print(f"   Center: {tf['center_timestamp']:.1f}s")
                print()
        
        return {
            "query": event_description,
            "frame_results": frame_results,
            "timeframe_results": timeframe_results,
            "total_found": len(frame_results)
        }
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return {"error": str(e)}

async def analyze_and_discuss_video_enhanced(
    video_path: str,
    subtitle_path: str = None,
    max_frames: int = 10,
    fps_extract: float = 0.2,
    discussion_rounds: int = 3,
    output_dir: str = "analysis_output",
    enable_vector_search: bool = True
):
    """Enhanced version of the original analyze_and_discuss_video function"""
    
    pipeline = EnhancedVideoAnalysisPipeline(enable_vector_search=enable_vector_search)
    
    return await pipeline.analyze_and_index_video(
        video_path=video_path,
        subtitle_path=subtitle_path,
        max_frames=max_frames,
        fps_extract=fps_extract,
        discussion_rounds=discussion_rounds,
        output_dir=output_dir
    )

def main():
    """Enhanced main entry point with vector search options"""
    parser = argparse.ArgumentParser(
        description="Analyze a video, generate AI agent discussion, and enable semantic search"
    )
    
    parser.add_argument(
        "video",
        help="Path to video file"
    )
    
    parser.add_argument(
        "--subtitles", "-s",
        help="Path to subtitle file (VTT or SRT)",
        default=None
    )
    
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
        help="Frame extraction rate in fps (default: 0.2 = 1 frame every 5 seconds)"
    )
    
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="Number of discussion rounds (default: 3)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="analysis_output",
        help="Output directory (default: analysis_output)"
    )
    
    parser.add_argument(
        "--no-vector-search",
        action="store_true",
        help="Disable vector search functionality"
    )
    
    parser.add_argument(
        "--search",
        help="Search for specific events in already indexed videos (e.g. 'person walking')"
    )
    
    args = parser.parse_args()
    
    # Handle search mode
    if args.search:
        print("🔍 SEARCH MODE")
        print("="*50)
        results = search_video_events(
            event_description=args.search,
            video_path=args.video if os.path.exists(args.video) else None
        )
        return
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    # Check if subtitle exists (if provided)
    if args.subtitles and not os.path.exists(args.subtitles):
        print(f"⚠️ Warning: Subtitle file not found: {args.subtitles}")
        print("Continuing without subtitles...")
        args.subtitles = None
    
    # Check vector search dependencies
    enable_vector_search = not args.no_vector_search
    if enable_vector_search:
        try:
            import pinecone
            import sentence_transformers
            if not os.getenv("PINECONE_API_KEY"):
                print("⚠️ Warning: PINECONE_API_KEY not found, disabling vector search")
                enable_vector_search = False
        except ImportError as e:
            print(f"⚠️ Warning: Vector search dependencies missing: {e}")
            print("Install with: pip install pinecone-client sentence-transformers")
            enable_vector_search = False
    
    # Run the enhanced pipeline
    asyncio.run(analyze_and_discuss_video_enhanced(
        video_path=args.video,
        subtitle_path=args.subtitles,
        max_frames=args.frames,
        fps_extract=args.fps,
        discussion_rounds=args.rounds,
        output_dir=args.output,
        enable_vector_search=enable_vector_search
    ))

if __name__ == "__main__":
    main()