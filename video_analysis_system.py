"""
Video Analysis System
Combines video processing with Fireworks.ai analysis
"""

import os
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time

from video_processor import VideoProcessor, SubtitleProcessor, VideoFrame, SubtitleSegment
from fireworks_client import FireworksClient, AnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VideoAnalysisResult:
    """Complete video analysis results"""
    video_path: str
    frame_count: int
    subtitle_count: int
    frame_analyses: List[Dict]
    subtitle_analyses: List[Dict]
    overall_analysis: str
    processing_time: float
    total_cost: float
    timestamp: str

class VideoAnalysisSystem:
    """Main system for video analysis using Fireworks.ai"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the video analysis system
        
        Args:
            api_key: Fireworks API key (optional, uses env var if not provided)
        """
        self.video_processor = VideoProcessor(output_dir="frames")
        self.subtitle_processor = SubtitleProcessor()
        self.fireworks_client = FireworksClient(api_key)
        
        logger.info("✅ Video Analysis System initialized")
    
    async def analyze_video(self,
                           video_path: str,
                           subtitle_path: Optional[str] = None,
                           max_frames: int = 10,
                           fps_extract: float = 0.2) -> VideoAnalysisResult:
        """Analyze a video with frames and subtitles
        
        Args:
            video_path: Path to video file
            subtitle_path: Optional path to subtitle file
            max_frames: Maximum frames to analyze
            fps_extract: Frames per second to extract
            
        Returns:
            VideoAnalysisResult with complete analysis
        """
        start_time = time.time()
        logger.info(f"Starting video analysis for: {video_path}")
        
        # Step 1: Extract frames
        logger.info("Step 1: Extracting frames...")
        frames = self.video_processor.extract_frames(
            video_path=video_path,
            fps_extract=fps_extract,
            max_frames=max_frames
        )
        logger.info(f"✅ Extracted {len(frames)} frames")
        
        # Step 2: Process subtitles if available
        subtitles = []
        if subtitle_path and os.path.exists(subtitle_path):
            logger.info("Step 2: Processing subtitles...")
            if subtitle_path.endswith('.vtt'):
                subtitles = self.subtitle_processor.parse_vtt(subtitle_path)
            elif subtitle_path.endswith('.srt'):
                subtitles = self.subtitle_processor.parse_srt(subtitle_path)
            logger.info(f"✅ Processed {len(subtitles)} subtitle segments")
        else:
            logger.info("Step 2: No subtitles provided")
        
        # Step 3: Analyze frames with Llama4 Maverick
        logger.info("Step 3: Analyzing frames with Llama4 Maverick...")
        frame_analyses = await self._analyze_frames(frames)
        
        # Step 4: Analyze subtitles
        subtitle_analyses = []
        if subtitles:
            logger.info("Step 4: Analyzing subtitles with AI...")
            subtitle_analyses = await self._analyze_subtitles(subtitles)
        else:
            logger.info("Step 4: Skipping subtitle analysis (no subtitles)")
        
        # Step 5: Generate overall analysis with GPT-OSS
        logger.info("Step 5: Generating overall video analysis with GPT-OSS...")
        overall_analysis = await self._generate_overall_analysis(
            frames=frames,
            frame_analyses=frame_analyses,
            subtitles=subtitles,
            subtitle_analyses=subtitle_analyses
        )
        
        # Calculate processing time and cost
        processing_time = time.time() - start_time
        usage_summary = self.fireworks_client.get_usage_summary()
        
        # Compile results
        result = VideoAnalysisResult(
            video_path=video_path,
            frame_count=len(frames),
            subtitle_count=len(subtitles),
            frame_analyses=frame_analyses,
            subtitle_analyses=subtitle_analyses,
            overall_analysis=overall_analysis,
            processing_time=processing_time,
            total_cost=usage_summary['total_cost_usd'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Analysis complete in {processing_time:.2f} seconds")
        logger.info(f"💰 Total cost: ${usage_summary['total_cost_usd']:.4f}")
        
        return result
    
    async def _analyze_frames(self, frames: List[VideoFrame]) -> List[Dict]:
        """Analyze video frames using Llama4 Maverick vision"""
        analyses = []
        
        for i, frame in enumerate(frames):
            logger.info(f"  Analyzing frame {i+1}/{len(frames)} at {frame.timestamp:.2f}s...")
            
            prompt = f"""Analyze this video frame at timestamp {frame.timestamp:.2f} seconds.
Describe:
1. Main subjects or objects visible
2. Actions or events happening
3. Setting or environment
4. Visual style and mood
5. Any text or important details

Be concise but specific."""
            
            try:
                # Use Llama4 Maverick for vision analysis
                result = await self.fireworks_client.analyze_frame(
                    base64_image=frame.base64_image,
                    prompt=prompt,
                    max_tokens=300
                )
                
                analyses.append({
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp,
                    "analysis": result.content,
                    "tokens_used": result.tokens_used,
                    "cost": result.cost
                })
                
            except Exception as e:
                logger.warning(f"  Frame analysis failed: {e}")
                # Add placeholder analysis
                analyses.append({
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp,
                    "analysis": f"Frame at {frame.timestamp:.2f}s - Analysis failed",
                    "tokens_used": 0,
                    "cost": 0
                })
        
        return analyses
    
    async def _analyze_subtitles(self, subtitles: List[SubtitleSegment]) -> List[Dict]:
        """Analyze subtitle segments using GPT-OSS"""
        analyses = []
        
        # Group subtitles into chunks for context
        chunk_size = 5
        for i in range(0, len(subtitles), chunk_size):
            chunk = subtitles[i:i+chunk_size]
            
            # Combine subtitle text
            combined_text = "\n".join([
                f"[{s.start_time:.1f}s - {s.end_time:.1f}s]: {s.text}"
                for s in chunk
            ])
            
            logger.info(f"  Analyzing subtitle chunk {i//chunk_size + 1}...")
            
            prompt = f"""Analyze this dialogue/text from the video:

{combined_text}

Identify:
1. Main topics discussed
2. Tone and emotion
3. Key information conveyed
4. Speaker characteristics (if apparent)

Be concise but insightful."""
            
            try:
                # Use GPT-OSS for subtitle analysis
                result = await self.fireworks_client.analyze_text(
                    text=prompt,
                    model_type="gpt_oss",
                    max_tokens=200
                )
                
                analyses.append({
                    "subtitle_range": f"{chunk[0].start_time:.1f}s - {chunk[-1].end_time:.1f}s",
                    "text_analyzed": combined_text,
                    "analysis": result.content,
                    "tokens_used": result.tokens_used,
                    "cost": result.cost
                })
                
            except Exception as e:
                logger.warning(f"  Subtitle analysis failed: {e}")
                # Try with smaller model as fallback
                try:
                    result = await self.fireworks_client.analyze_text(
                        text=prompt,
                        model_type="small",
                        max_tokens=150
                    )
                    analyses.append({
                        "subtitle_range": f"{chunk[0].start_time:.1f}s - {chunk[-1].end_time:.1f}s",
                        "text_analyzed": combined_text,
                        "analysis": result.content,
                        "tokens_used": result.tokens_used,
                        "cost": result.cost
                    })
                except:
                    analyses.append({
                        "subtitle_range": f"{chunk[0].start_time:.1f}s - {chunk[-1].end_time:.1f}s",
                        "text_analyzed": combined_text,
                        "analysis": "Analysis failed",
                        "tokens_used": 0,
                        "cost": 0
                    })
        
        return analyses
    
    async def _generate_overall_analysis(self,
                                        frames: List[VideoFrame],
                                        frame_analyses: List[Dict],
                                        subtitles: List[SubtitleSegment],
                                        subtitle_analyses: List[Dict]) -> str:
        """Generate comprehensive overall analysis using GPT-OSS"""
        
        # Prepare summary of frame analyses
        frame_summary = ""
        if frame_analyses:
            frame_points = []
            for fa in frame_analyses[:5]:  # First 5 frames
                frame_points.append(f"- At {fa['timestamp']:.1f}s: {fa['analysis'][:100]}...")
            frame_summary = "Visual Analysis:\n" + "\n".join(frame_points)
        
        # Prepare summary of subtitle analyses
        subtitle_summary = ""
        if subtitle_analyses:
            sub_points = []
            for sa in subtitle_analyses[:3]:  # First 3 chunks
                sub_points.append(f"- {sa['subtitle_range']}: {sa['analysis'][:100]}...")
            subtitle_summary = "Dialogue/Audio Analysis:\n" + "\n".join(sub_points)
        
        # Create comprehensive prompt
        prompt = f"""Based on the analysis of this video, provide a comprehensive summary.

Video Information:
- Duration: Approximately {frames[-1].timestamp if frames else 0:.1f} seconds
- Frames analyzed: {len(frames)}
- Subtitle segments: {len(subtitles)}

{frame_summary}

{subtitle_summary}

Please provide:
1. Overall description of the video content
2. Main themes or messages
3. Visual style and production quality
4. Key moments or highlights
5. Suggested improvements or observations

Be thorough but concise (max 300 words)."""
        
        try:
            # Try GPT-OSS first for best quality
            result = await self.fireworks_client.analyze_text(
                text=prompt,
                model_type="gpt_oss",
                max_tokens=400
            )
            return result.content
            
        except Exception as e:
            logger.error(f"GPT-OSS analysis failed: {e}, trying fallback...")
            # Fallback to smaller model
            try:
                result = await self.fireworks_client.analyze_text(
                    text=prompt,
                    model_type="small",
                    max_tokens=300
                )
                return result.content
            except:
                return "Overall analysis could not be generated due to API errors."
    
    def save_results(self, results: VideoAnalysisResult, output_path: str = "video_analysis_results.json"):
        """Save analysis results to JSON file"""
        # Convert dataclass to dict
        results_dict = asdict(results)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"✅ Results saved to {output_path}")
    
    def print_summary(self, results: VideoAnalysisResult):
        """Print a summary of the analysis"""
        print("\n" + "="*60)
        print("VIDEO ANALYSIS SUMMARY")
        print("="*60)
        print(f"Video: {results.video_path}")
        print(f"Frames analyzed: {results.frame_count}")
        print(f"Subtitles processed: {results.subtitle_count}")
        print(f"Processing time: {results.processing_time:.2f} seconds")
        print(f"Total cost: ${results.total_cost:.4f}")
        print("\n" + "-"*40)
        print("OVERALL ANALYSIS:")
        print("-"*40)
        print(results.overall_analysis)
        print("="*60)


# Test the system
async def test_video_analysis():
    """Test the complete video analysis system"""
    print("="*60)
    print("VIDEO ANALYSIS SYSTEM TEST")
    print("="*60)
    
    # Initialize system
    try:
        system = VideoAnalysisSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Create test video and subtitles
    test_video = "test_videos/sample.mp4"
    test_subtitles = "test_videos/sample.vtt"
    
    # Make sure we have test files
    os.makedirs("test_videos", exist_ok=True)
    
    # Create test subtitle if not exists
    if not os.path.exists(test_subtitles):
        with open(test_subtitles, 'w') as f:
            f.write("""WEBVTT

00:00:00.000 --> 00:00:05.000
Welcome to this test video demonstration.

00:00:05.000 --> 00:00:10.000
This video shows our analysis system in action.

00:00:10.000 --> 00:00:15.000
The system can analyze both visual and audio content.
""")
    
    # Run analysis
    try:
        results = await system.analyze_video(
            video_path=test_video,
            subtitle_path=test_subtitles,
            max_frames=3,  # Analyze only 3 frames for testing
            fps_extract=0.2
        )
        
        # Print summary
        system.print_summary(results)
        
        # Save results
        system.save_results(results, "test_analysis_results.json")
        
        print("\n✅ Video analysis test complete!")
        print(f"📄 Full results saved to test_analysis_results.json")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_video_analysis())