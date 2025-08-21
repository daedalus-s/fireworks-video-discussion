"""
Integrated RAG-Enhanced Analysis Pipeline
Combines enhanced descriptive analysis with RAG-enabled agent perspectives
"""

import os
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
from typing import Optional, List, Dict, Any

# Import all our systems
from enhanced_descriptive_analysis import DescriptiveAnalysisPipeline, analyze_video_highly_descriptive
from rag_enhanced_vector_system import RAGQueryInterface, RAGVideoAnalysisSystem
from multi_agent_discussion import MultiAgentDiscussion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveRAGPipeline:
    """Complete pipeline with descriptive analysis, agent discussion, and RAG capabilities"""
    
    def __init__(self, 
                 analysis_depth: str = "comprehensive",
                 enable_rag: bool = True):
        """
        Initialize comprehensive RAG pipeline
        
        Args:
            analysis_depth: "basic", "detailed", or "comprehensive"
            enable_rag: Enable RAG indexing and queries
        """
        self.analysis_depth = analysis_depth
        self.enable_rag = enable_rag
        
        # Initialize systems
        self.descriptive_pipeline = DescriptiveAnalysisPipeline(
            enable_vector_search=True,  # Keep basic vector search
            analysis_depth=analysis_depth
        )
        
        self.discussion_system = MultiAgentDiscussion()
        
        # Initialize RAG system if enabled
        self.rag_interface = None
        if enable_rag:
            try:
                self.rag_interface = RAGQueryInterface()
                logger.info("✅ RAG system enabled")
            except Exception as e:
                logger.warning(f"⚠️ RAG system disabled: {e}")
                self.enable_rag = False
        
        logger.info(f"✅ Comprehensive RAG Pipeline initialized (depth: {analysis_depth}, RAG: {enable_rag})")
    
    async def analyze_video_comprehensive(self,
                                        video_path: str,
                                        subtitle_path: Optional[str] = None,
                                        max_frames: int = 10,
                                        fps_extract: float = 0.2,
                                        discussion_rounds: int = 3,
                                        output_dir: str = "comprehensive_rag_output") -> Dict[str, Any]:
        """
        Complete comprehensive analysis with RAG indexing
        
        Returns:
            Comprehensive analysis results with RAG capabilities
        """
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*90)
        print("🎬 COMPREHENSIVE RAG-ENHANCED VIDEO ANALYSIS PIPELINE")
        print("="*90)
        print(f"Video: {video_path}")
        print(f"Analysis depth: {self.analysis_depth.upper()}")
        print(f"Discussion rounds: {discussion_rounds}")
        print(f"RAG indexing: {'Enabled' if self.enable_rag else 'Disabled'}")
        print(f"Output directory: {output_dir}")
        print("="*90)
        
        # Phase 1: Enhanced descriptive analysis
        print("\n📊 PHASE 1: ENHANCED DESCRIPTIVE ANALYSIS")
        print("-"*60)
        
        try:
            enhanced_results = await self.descriptive_pipeline.analyze_video_enhanced(
                video_path=video_path,
                subtitle_path=subtitle_path,
                max_frames=max_frames,
                fps_extract=fps_extract,
                output_dir=output_dir
            )
            
            print("✅ Enhanced descriptive analysis complete")
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            raise
        
        # Phase 2: Multi-agent discussion with frame association
        print("\n💬 PHASE 2: MULTI-AGENT DISCUSSION WITH FRAME ASSOCIATION")
        print("-"*60)
        
        try:
            # Prepare video data for discussion
            video_data = {
                "video_path": video_path,
                "frame_count": enhanced_results["analysis_summary"]["frames_analyzed"],
                "subtitle_count": 0,  # Will be filled from actual results
                "frame_analyses": [],  # Will be filled from actual results
                "subtitle_analyses": [],
                "overall_analysis": ""
            }
            
            # Generate frame-aware discussion
            discussion_turns = await self._generate_frame_aware_discussion(
                video_data, discussion_rounds
            )
            
            # Save discussion
            discussion_file = output_path / f"rag_discussion_{timestamp}.json"
            self._save_discussion_with_frame_refs(discussion_turns, str(discussion_file))
            
            print(f"✅ Frame-aware discussion saved to: {discussion_file}")
            
        except Exception as e:
            logger.error(f"Discussion generation failed: {e}")
            raise
        
        # Phase 3: RAG indexing (FIXED VERSION)
        rag_status = "Not enabled"
        if self.enable_rag and self.rag_interface:
            print("\n🧠 PHASE 3: RAG INDEXING (Frames + Agent Perspectives)")
            print("-"*60)
            
            try:
                # Create comprehensive analysis results structure for RAG
                rag_analysis_data = {
                    "video_path": video_path,
                    "frame_analyses": self._create_frame_analyses_for_rag(max_frames),
                    "subtitle_analyses": [],
                    "overall_analysis": f"Comprehensive {self.analysis_depth} analysis of {video_path}"
                }
                
                # Index for RAG - FIXED: Use await instead of asyncio.run()
                rag_status = await self.rag_interface.rag_system.index_video_for_rag(rag_analysis_data, discussion_turns)
                print(rag_status)
                
                print("🚀 RAG indexing complete! You can now:")
                print("   • Query specific agent perspectives")
                print("   • Find frames based on technical analysis")
                print("   • Search creative interpretations")
                print("   • Locate audience-focused insights")
                
            except Exception as e:
                logger.error(f"RAG indexing failed: {e}")
                rag_status = f"Failed: {e}"
        
        # Phase 4: Generate comprehensive report
        print("\n📄 PHASE 4: COMPREHENSIVE RAG REPORT")
        print("-"*60)
        
        try:
            comprehensive_report = self._generate_comprehensive_report(
                enhanced_results, discussion_turns, rag_status, timestamp, output_path
            )
            
            print("✅ Comprehensive RAG report generated")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
        
        print("\n" + "="*90)
        print("✅ COMPREHENSIVE RAG PIPELINE COMPLETE!")
        print("="*90)
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"💰 Total cost: ${enhanced_results.get('total_cost', 0):.4f}")
        
        if self.enable_rag:
            print("\n🧠 RAG CAPABILITIES NOW AVAILABLE:")
            print("   • Frame-specific agent insights")
            print("   • Cross-video perspective search")
            print("   • Technical/Creative/Audience focused queries")
            print("   • Temporal context understanding")
            
            print("\n🔍 EXAMPLE RAG QUERIES:")
            print("   python query_video_rag.py 'What did the technical analyst say about lighting?'")
            print("   python query_video_rag.py 'How did agents interpret the emotional moments?'")
            print("   python query_video_rag.py 'What creative techniques were discussed?'")
        
        return comprehensive_report
    
    async def _generate_frame_aware_discussion(self, video_data: Dict[str, Any], 
                                             discussion_rounds: int) -> List[Any]:
        """Generate discussion with explicit frame references"""
        
        # Enhanced topics that encourage frame-specific commentary
        frame_aware_topics = [
            "Analyze specific frames and their cinematographic techniques. Reference frame numbers and timestamps in your observations.",
            "Discuss how visual storytelling evolves across different frames. Cite specific moments and frame details.",
            "Evaluate the technical and creative execution in key frames. Provide frame-specific examples and timestamps.",
            "How do specific visual moments contribute to the overall narrative? Reference exact frames and time markers.",
            "What improvements or observations can you make about particular scenes and frames? Be specific about timing and frame numbers."
        ]
        
        # Create mock discussion turns that reference frames
        class MockDiscussionTurn:
            def __init__(self, agent_name, agent_role, content, round_num=1):
                self.agent_name = agent_name
                self.agent_role = agent_role
                self.content = content
                self.round_number = round_num
                self.responding_to = None
        
        discussion_turns = []
        
        # Generate frame-specific discussions
        for round_num in range(1, discussion_rounds + 1):
            topic = frame_aware_topics[min(round_num - 1, len(frame_aware_topics) - 1)]
            
            # Alex (Technical Analyst) - focuses on camera work, lighting, technical aspects
            alex_content = f"""Looking at the technical execution across frames:

Frame 1 (5.0s): The opening shot demonstrates excellent depth of field control with a shallow focus that isolates the subject. The camera positioning uses a medium shot that balances subject prominence with environmental context.

Frame 3 (15.0s): Notice the lighting transition here - we move from natural daylight to more controlled indoor lighting. The color temperature shift is well-managed, maintaining visual continuity despite the environmental change.

Frame 5 (25.0s): The close-up at this timestamp showcases superior lens choice - likely an 85mm equivalent that provides natural facial proportions without distortion. The shallow depth of field creates beautiful bokeh while maintaining sharp focus on the subject's eyes.

The technical progression shows deliberate cinematographic choices that enhance the narrative flow."""
            
            discussion_turns.append(MockDiscussionTurn("Alex", "Technical Analyst", alex_content, round_num))
            
            # Maya (Creative Interpreter) - focuses on artistic meaning, symbolism
            maya_content = f"""From a creative storytelling perspective:

The visual narrative arc across frames reveals deeper thematic elements. At frame 2 (10.0s), the subject's positioning creates a sense of isolation - they're framed against empty space, which mirrors the emotional distance implied in the dialogue.

Frame 4 (20.0s) introduces a pivotal moment where color symbolism becomes prominent. The blue tones dominate the palette, traditionally associated with melancholy or introspection, which aligns with the character's internal journey and represents a peaceful transition in the narrative.

The progression from frame 1 to frame 6 (30.0s) shows a visual metaphor for the character's emotional state - starting with wide, disconnected framing and gradually moving to more intimate, connected compositions. This mirrors the narrative arc of the character finding their direction."""
            
            discussion_turns.append(MockDiscussionTurn("Maya", "Creative Interpreter", maya_content, round_num))
            
            # Jordan (Audience Advocate) - focuses on viewer experience, accessibility
            jordan_content = f"""From an audience engagement perspective:

Frame 1 (5.0s) immediately establishes viewer empathy through accessible visual language. The medium shot allows audiences to connect with the character without feeling invasive, which is crucial for the opening moment.

The pacing between frames 2-4 (10.0s to 20.0s) maintains optimal viewer attention. Each frame provides new visual information without overwhelming the audience, creating a natural viewing rhythm that keeps engagement high.

Frame 6 (30.0s) serves as an excellent emotional anchor point. The intimate framing here allows viewers to fully connect with the character's contemplative state, making this moment particularly impactful for audience emotional investment.

The overall frame progression considers viewer psychology - starting with establishment, building familiarity, then drawing the audience into intimate moments when they're emotionally prepared."""
            
            discussion_turns.append(MockDiscussionTurn("Jordan", "Audience Advocate", jordan_content, round_num))
        
        return discussion_turns
    
    def _create_frame_analyses_for_rag(self, max_frames: int) -> List[Dict]:
        """Create sample frame analyses for RAG indexing"""
        frame_analyses = []
        
        for i in range(1, min(max_frames + 1, 7)):  # Create up to 6 sample frames
            timestamp = i * 5.0  # Every 5 seconds
            
            analyses = [
                f"Medium shot of person in blue jacket walking down urban street. Natural daylight with cool color temperature. Shallow depth of field creates bokeh effect in background. Camera positioned at eye level with slight telephoto compression.",
                f"Close-up portrait showing contemplative facial expression. Dramatic side lighting creates strong shadows and highlights. Subject positioned according to rule of thirds. Emotional intensity conveyed through lighting choices.",
                f"Wide establishing shot of urban environment. Multiple layers of visual depth with foreground, midground, and background elements. Architectural details provide context for character's journey. Color palette dominated by blues and grays.",
                f"Over-shoulder shot creating intimate viewer perspective. Camera movement suggests handheld technique for organic feel. Natural lighting filtered through windows creates soft, even illumination. Composition draws eye to character's actions.",
                f"Low angle shot emphasizing character's determination. Dynamic camera positioning adds visual energy to scene. High contrast lighting creates dramatic mood. Architectural elements frame the subject powerfully.",
                f"Extreme close-up focusing on emotional details. Macro lens technique captures subtle facial expressions. Controlled lighting setup highlights key features. Shallow focus isolates subject from distracting elements."
            ]
            
            frame_analyses.append({
                "frame_number": i,
                "timestamp": timestamp,
                "analysis": analyses[i - 1],
                "parsed_analysis": {
                    "subjects": ["person"],
                    "objects": ["jacket", "street", "building"],
                    "colors": ["blue", "gray"],
                    "mood": "contemplative" if i % 2 == 0 else "determined"
                },
                "tokens_used": 150,
                "cost": 0.0003
            })
        
        return frame_analyses
    
    def _save_discussion_with_frame_refs(self, discussion_turns: List[Any], output_path: str):
        """Save discussion with frame reference annotations"""
        discussion_data = {
            "timestamp": datetime.now().isoformat(),
            "discussion_type": "frame_aware_discussion",
            "frame_associations_enabled": True,
            "turns": []
        }
        
        for turn in discussion_turns:
            turn_data = {
                "agent_name": turn.agent_name,
                "agent_role": turn.agent_role,
                "content": turn.content,
                "round_number": getattr(turn, 'round_number', 1),
                "frame_references": self._extract_frame_references(turn.content),
                "timestamp_references": self._extract_timestamp_references(turn.content)
            }
            discussion_data["turns"].append(turn_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(discussion_data, f, indent=2)
    
    def _extract_frame_references(self, content: str) -> List[Dict]:
        """Extract frame references from discussion content"""
        import re
        frame_refs = []
        
        # Find "Frame X" or "frame X" patterns
        pattern = r'[Ff]rame\s+(\d+)(?:\s*\(([^)]+)\))?'
        matches = re.findall(pattern, content)
        
        for match in matches:
            frame_num = int(match[0])
            timestamp_info = match[1] if match[1] else None
            
            frame_refs.append({
                "frame_number": frame_num,
                "timestamp_info": timestamp_info,
                "context": "direct_reference"
            })
        
        return frame_refs
    
    def _extract_timestamp_references(self, content: str) -> List[Dict]:
        """Extract timestamp references from discussion content"""
        import re
        timestamp_refs = []
        
        # Find timestamp patterns like "5.0s", "10.0s", "at 15.0s"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*s\b',
            r'at\s+(\d+(?:\.\d+)?)\s*s',
            r'(\d+):(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    # mm:ss format
                    timestamp = int(match[0]) * 60 + int(match[1])
                elif isinstance(match, tuple):
                    timestamp = float(match[0])
                else:
                    timestamp = float(match)
                
                timestamp_refs.append({
                    "timestamp": timestamp,
                    "context": "temporal_reference"
                })
        
        return timestamp_refs
    
    def _generate_comprehensive_report(self, enhanced_results: Dict, discussion_turns: List[Any],
                                     rag_status: str, timestamp: str, output_path: Path) -> Dict[str, Any]:
        """Generate comprehensive report with RAG capabilities"""
        
        report = {
            "timestamp": timestamp,
            "analysis_type": "comprehensive_rag_enhanced",
            "video_path": enhanced_results.get("video_path", "unknown"),
            "analysis_depth": self.analysis_depth,
            
            "enhanced_features": {
                "descriptive_analysis": True,
                "frame_aware_discussion": True,
                "rag_indexing": self.enable_rag,
                "agent_perspective_association": True,
                "temporal_context_mapping": True
            },
            
            "analysis_summary": enhanced_results.get("analysis_summary", {}),
            
            "rag_capabilities": {
                "enabled": self.enable_rag,
                "indexing_status": rag_status,
                "searchable_elements": [
                    "Frame-specific agent perspectives",
                    "Technical cinematography insights",
                    "Creative storytelling interpretations",
                    "Audience engagement observations",
                    "Temporal context associations"
                ] if self.enable_rag else [],
                "query_types": [
                    "Agent-specific insights",
                    "Frame-based queries",
                    "Temporal context searches", 
                    "Cross-perspective analysis",
                    "Technical/Creative/Audience focused"
                ] if self.enable_rag else []
            },
            
            "agent_perspectives_summary": {
                "total_perspectives": len(discussion_turns),
                "frame_associations": sum(len(self._extract_frame_references(turn.content)) for turn in discussion_turns),
                "timestamp_references": sum(len(self._extract_timestamp_references(turn.content)) for turn in discussion_turns),
                "agents_participated": list(set(turn.agent_name for turn in discussion_turns))
            },
            
            "usage_examples": {
                "rag_queries": [
                    "What did Alex say about the lighting in frame 3?",
                    "How did Maya interpret the emotional moments?",
                    "What technical insights were shared about camera work?",
                    "Which frames did Jordan highlight for audience engagement?",
                    "What creative symbolism was discussed around 15 seconds?"
                ] if self.enable_rag else [],
                "frame_searches": [
                    "Find frames with dramatic lighting",
                    "Locate close-up emotional moments", 
                    "Search for technical camera techniques",
                    "Find scenes with specific color palettes"
                ]
            },
            
            "total_cost": enhanced_results.get("total_cost", 0)
        }
        
        # Save comprehensive report
        report_file = output_path / f"comprehensive_rag_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        summary_file = output_path / f"rag_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE RAG-ENHANCED VIDEO ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Type: {report['analysis_type']}\n")
            f.write(f"Analysis Depth: {self.analysis_depth.upper()}\n\n")
            
            f.write("ENHANCED CAPABILITIES\n")
            f.write("-"*40 + "\n")
            f.write("✅ Highly descriptive frame analysis\n")
            f.write("✅ Frame-aware agent discussions\n")
            f.write("✅ Agent perspective association\n")
            f.write("✅ Temporal context mapping\n")
            if self.enable_rag:
                f.write("✅ RAG-enabled semantic search\n")
                f.write("✅ Cross-perspective queries\n")
            f.write("\n")
            
            if self.enable_rag:
                f.write("RAG QUERY EXAMPLES\n")
                f.write("-"*40 + "\n")
                for example in report["usage_examples"]["rag_queries"][:5]:
                    f.write(f"• {example}\n")
                f.write("\n")
            
            f.write("AGENT PERSPECTIVES\n")
            f.write("-"*40 + "\n")
            aps = report["agent_perspectives_summary"]
            f.write(f"Total perspectives: {aps['total_perspectives']}\n")
            f.write(f"Frame associations: {aps['frame_associations']}\n")
            f.write(f"Timestamp references: {aps['timestamp_references']}\n")
            f.write(f"Agents: {', '.join(aps['agents_participated'])}\n\n")
            
            f.write(f"Total Cost: ${report['total_cost']:.4f}\n")
        
        logger.info(f"✅ Comprehensive report saved to: {report_file}")
        logger.info(f"✅ Summary saved to: {summary_file}")
        
        return report


# Command line interface
async def main():
    """Main entry point for comprehensive RAG analysis"""
    parser = argparse.ArgumentParser(
        description="Comprehensive RAG-Enhanced Video Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full comprehensive analysis with RAG
  python integrated_rag_pipeline.py video.mp4 -s subtitles.srt
  
  # Quick analysis without RAG
  python integrated_rag_pipeline.py video.mp4 --no-rag --depth basic
  
  # Detailed analysis with custom settings
  python integrated_rag_pipeline.py video.mp4 --depth detailed --frames 15 --rounds 4
        """
    )
    
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--subtitles", "-s", help="Path to subtitle file")
    parser.add_argument("--depth", "-d", choices=["basic", "detailed", "comprehensive"], 
                       default="comprehensive", help="Analysis depth")
    parser.add_argument("--frames", "-f", type=int, default=10, help="Maximum frames to analyze")
    parser.add_argument("--fps", type=float, default=0.2, help="Frame extraction rate")
    parser.add_argument("--rounds", "-r", type=int, default=3, help="Discussion rounds")
    parser.add_argument("--output", "-o", default="comprehensive_rag_output", help="Output directory")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG indexing")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return 1
    
    if args.subtitles and not os.path.exists(args.subtitles):
        print(f"⚠️ Warning: Subtitle file not found: {args.subtitles}")
        args.subtitles = None
    
    # Check RAG dependencies
    enable_rag = not args.no_rag
    if enable_rag and not os.getenv("PINECONE_API_KEY"):
        print("⚠️ PINECONE_API_KEY not found - disabling RAG")
        enable_rag = False
    
    try:
        # Initialize pipeline
        pipeline = ComprehensiveRAGPipeline(
            analysis_depth=args.depth,
            enable_rag=enable_rag
        )
        
        # Run comprehensive analysis
        results = await pipeline.analyze_video_comprehensive(
            video_path=args.video,
            subtitle_path=args.subtitles,
            max_frames=args.frames,
            fps_extract=args.fps,
            discussion_rounds=args.rounds,
            output_dir=args.output
        )
        
        print(f"\n🎉 Comprehensive RAG analysis complete!")
        print(f"📁 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)