"""
Complete Pipeline for Analyzing Real Videos
Processes video, analyzes content, and generates multi-agent discussion
"""

import os
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

from video_processor import VideoProcessor, SubtitleProcessor
from fireworks_client import FireworksClient
from video_analysis_system import VideoAnalysisSystem
from multi_agent_discussion import MultiAgentDiscussion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def analyze_and_discuss_video(
    video_path: str,
    subtitle_path: str = None,
    max_frames: int = 10,
    fps_extract: float = 0.2,
    discussion_rounds: int = 3,
    output_dir: str = "analysis_output"
):
    """
    Complete pipeline to analyze a video and generate discussion
    
    Args:
        video_path: Path to video file
        subtitle_path: Path to subtitle file (optional)
        max_frames: Maximum frames to analyze
        fps_extract: Frames per second to extract
        discussion_rounds: Number of discussion rounds
        output_dir: Directory for output files
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate timestamp for this analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*70)
    print("🎬 VIDEO ANALYSIS AND DISCUSSION PIPELINE")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Subtitles: {subtitle_path if subtitle_path else 'None'}")
    print(f"Max frames: {max_frames}")
    print(f"Frame extraction rate: {fps_extract} fps")
    print(f"Discussion rounds: {discussion_rounds}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # Step 1: Analyze the video
    print("\n📊 PHASE 1: VIDEO ANALYSIS")
    print("-"*40)
    
    try:
        analysis_system = VideoAnalysisSystem()
        
        video_results = await analysis_system.analyze_video(
            video_path=video_path,
            subtitle_path=subtitle_path,
            max_frames=max_frames,
            fps_extract=fps_extract
        )
        
        # Save video analysis
        analysis_file = output_path / f"video_analysis_{timestamp}.json"
        analysis_system.save_results(video_results, str(analysis_file))
        
        # Print summary
        analysis_system.print_summary(video_results)
        
        print(f"\n✅ Video analysis saved to: {analysis_file}")
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise
    
    # Step 2: Generate multi-agent discussion
    print("\n💬 PHASE 2: MULTI-AGENT DISCUSSION")
    print("-"*40)
    
    try:
        discussion_system = MultiAgentDiscussion()
        
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
        discussion = await discussion_system.discuss_video(
            video_analysis=video_data,
            num_rounds=discussion_rounds
        )
        
        # Print discussion
        discussion_system.print_discussion()
        
        # Save discussion
        discussion_file = output_path / f"discussion_{timestamp}.json"
        discussion_system.save_discussion(str(discussion_file))
        
        print(f"\n✅ Discussion saved to: {discussion_file}")
        
    except Exception as e:
        logger.error(f"Discussion generation failed: {e}")
        raise
    
    # Step 3: Generate combined report
    print("\n📄 PHASE 3: GENERATING FINAL REPORT")
    print("-"*40)
    
    try:
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
            "key_insights": {
                "visual_highlights": [
                    f"Frame {fa['timestamp']:.1f}s: {fa['analysis'][:150]}..."
                    for fa in video_results.frame_analyses[:3]
                ],
                "dialogue_highlights": [
                    f"{sa['subtitle_range']}: {sa['analysis'][:150]}..."
                    for sa in video_results.subtitle_analyses[:2]
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
            "total_cost": discussion_system.client.get_usage_summary()['total_cost_usd']
        }
        
        # Save report
        report_file = output_path / f"complete_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Also create a readable text report
        text_report_file = output_path / f"report_{timestamp}.txt"
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write("VIDEO ANALYSIS AND DISCUSSION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Subtitles: {subtitle_path if subtitle_path else 'None'}\n\n")
            
            f.write("ANALYSIS SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Frames analyzed: {video_results.frame_count}\n")
            f.write(f"Processing time: {video_results.processing_time:.2f} seconds\n")
            f.write(f"Overall assessment:\n{video_results.overall_analysis}\n\n")
            
            f.write("MULTI-AGENT DISCUSSION\n")
            f.write("-"*40 + "\n")
            for turn in discussion:
                f.write(f"\n{turn.agent_name} ({turn.agent_role}):\n")
                f.write(f"{turn.content}\n")
                f.write("-"*30 + "\n")
            
            f.write(f"\nTotal cost: ${report['total_cost']:.4f}\n")
        
        print(f"✅ Complete report saved to: {report_file}")
        print(f"✅ Text report saved to: {text_report_file}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise
    
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"💰 Total cost: ${report['total_cost']:.4f}")
    print("="*70)
    
    return report


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze a video and generate AI agent discussion"
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
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    # Check if subtitle exists (if provided)
    if args.subtitles and not os.path.exists(args.subtitles):
        print(f"⚠️ Warning: Subtitle file not found: {args.subtitles}")
        print("Continuing without subtitles...")
        args.subtitles = None
    
    # Run the pipeline
    asyncio.run(analyze_and_discuss_video(
        video_path=args.video,
        subtitle_path=args.subtitles,
        max_frames=args.frames,
        fps_extract=args.fps,
        discussion_rounds=args.rounds,
        output_dir=args.output
    ))


if __name__ == "__main__":
    main()