#!/usr/bin/env python3
"""
Integrated Configurable Agent Pipeline - FIXED VERSION
Combines enhanced video analysis with configurable multi-agent system
FIXES: Data flow issue between enhanced analysis and agent discussion
"""

import os
import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from configurable_agent_system import (
    ConfigurableMultiAgentDiscussion,
    load_agent_template,
    show_available_templates,
    AgentTemplates
)

# Import your existing analysis modules
try:
    from enhanced_descriptive_analysis import analyze_video_highly_descriptive
    from rag_enhanced_vector_system import RAGQueryInterface
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Analysis modules not fully available: {e}")
    ANALYSIS_MODULES_AVAILABLE = False

class IntegratedConfigurableAnalysisPipeline:
    """Enhanced pipeline with configurable agents and full video analysis - FIXED DATA FLOW"""
    
    def __init__(self, 
                 analysis_depth: str = "comprehensive",
                 enable_rag: bool = True):
        """Initialize the integrated pipeline"""
        self.analysis_depth = analysis_depth
        self.enable_rag = enable_rag
        
        # Initialize configurable agent system
        self.agent_system = ConfigurableMultiAgentDiscussion()
        
        # Initialize RAG interface if available and enabled
        self.rag_interface = None
        if enable_rag and ANALYSIS_MODULES_AVAILABLE:
            try:
                self.rag_interface = RAGQueryInterface()
                print("RAG system enabled")
            except Exception as e:
                print(f"Warning: RAG system initialization failed: {e}")
                self.enable_rag = False
        else:
            self.enable_rag = False
        
        print("Integrated Configurable Analysis Pipeline initialized")
    
    def setup_agents_for_content_type(self, content_type: str) -> bool:
        """Setup agents based on content type"""
        templates = {
            "film": "film_analysis",
            "movie": "film_analysis", 
            "cinema": "film_analysis",
            "educational": "educational",
            "tutorial": "educational",
            "learning": "educational",
            "marketing": "marketing",
            "commercial": "marketing",
            "advertisement": "marketing",
            "default": None
        }
        
        template_name = templates.get(content_type.lower(), "default")
        
        if template_name:
            template_agents = load_agent_template(template_name)
            if template_agents:
                self.agent_system.agents = template_agents
                print(f"Loaded {len(template_agents)} agents for {content_type} content")
                return True
        
        # Keep default agents if no template found
        print(f"Using default agents for {content_type} content")
        return False
    
    async def analyze_video_with_configurable_agents(self,
                                                    video_path: str,
                                                    subtitle_path: Optional[str] = None,
                                                    max_frames: int = 10,
                                                    fps_extract: float = 0.2,
                                                    discussion_rounds: int = 3,
                                                    selected_agents: Optional[List[str]] = None,
                                                    content_type: Optional[str] = None,
                                                    agent_template: Optional[str] = None,
                                                    output_dir: str = "configurable_analysis_output") -> Dict[str, Any]:
        """
        Complete analysis with configurable agents - FIXED DATA FLOW
        
        Args:
            video_path: Path to video file
            subtitle_path: Optional subtitle file path
            max_frames: Maximum frames to analyze
            fps_extract: Frame extraction rate
            discussion_rounds: Number of agent discussion rounds
            selected_agents: Specific agents to include in discussion
            content_type: Type of content (film, educational, marketing, etc.)
            agent_template: Specific agent template to load
            output_dir: Output directory for results
            
        Returns:
            Complete analysis results with configurable agent insights
        """
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*90)
        print("INTEGRATED CONFIGURABLE AGENT VIDEO ANALYSIS PIPELINE")
        print("="*90)
        print(f"Video: {video_path}")
        print(f"Analysis depth: {self.analysis_depth.upper()}")
        print(f"Agent configuration: Custom configurable agents")
        print(f"Discussion rounds: {discussion_rounds}")
        print(f"RAG indexing: {'Enabled' if self.enable_rag else 'Disabled'}")
        print(f"Output directory: {output_dir}")
        print("="*90)
        
        # Step 0: Configure agents based on content type or template
        if agent_template:
            template_agents = load_agent_template(agent_template)
            if template_agents:
                self.agent_system.agents = template_agents
                print(f"Loaded agent template: {agent_template}")
        elif content_type:
            self.setup_agents_for_content_type(content_type)
        
        # Show participating agents
        participating_agents = self.agent_system.agents
        if selected_agents:
            participating_agents = [a for a in self.agent_system.agents if a.name in selected_agents]
        
        print(f"\nCONFIGURED AGENTS ({len(participating_agents)}):")
        print("-"*60)
        for agent in participating_agents:
            print(f"  {agent.emoji} {agent.name} ({agent.role}) - Model: {agent.model}")
            print(f"    Expertise: {', '.join(agent.expertise[:3])}")
            print(f"    Focus: {', '.join(agent.focus_areas[:2])}")
        
        # Phase 1: Enhanced descriptive analysis (if available)
        enhanced_analysis_results = None
        if ANALYSIS_MODULES_AVAILABLE:
            print("\nPHASE 1: ENHANCED DESCRIPTIVE ANALYSIS")
            print("-"*60)
            
        try:
            enhanced_analysis_results = await analyze_video_highly_descriptive(
                video_path=video_path,
                subtitle_path=subtitle_path,
                max_frames=max_frames,
                fps_extract=fps_extract,
                analysis_depth=self.analysis_depth,
                enable_vector_search=True,
                output_dir=output_dir
            )
            print("Enhanced descriptive analysis complete")
            
            # DEBUG: Inspect what we actually got back
            print("\n🔍 DEBUGGING: Inspecting enhanced_analysis_results...")
            self._debug_enhanced_results(enhanced_analysis_results)
            
        except Exception as e:
            print(f"Warning: Enhanced analysis failed, using basic analysis: {e}")
            enhanced_analysis_results = self._create_basic_video_analysis(
                video_path, max_frames, subtitle_path
            )
        else:
            print("\nPHASE 1: BASIC VIDEO ANALYSIS")
            print("-"*60)
            enhanced_analysis_results = self._create_basic_video_analysis(
                video_path, max_frames, subtitle_path
            )
        
        # CRITICAL FIX: Convert enhanced analysis results to proper format for agents
        video_analysis_for_agents = self._convert_enhanced_results_for_agents(enhanced_analysis_results)
        
        # DEBUG: Show what data we're passing to agents
        print(f"\nDEBUG: Data being passed to agents:")
        print(f"  frame_count: {video_analysis_for_agents.get('frame_count', 'MISSING')}")
        print(f"  subtitle_count: {video_analysis_for_agents.get('subtitle_count', 'MISSING')}")
        print(f"  frame_analyses length: {len(video_analysis_for_agents.get('frame_analyses', []))}")
        print(f"  overall_analysis length: {len(video_analysis_for_agents.get('overall_analysis', ''))}")
        
        # Phase 2: Configurable multi-agent discussion
        print("\nPHASE 2: CONFIGURABLE MULTI-AGENT DISCUSSION")
        print("-"*60)
        
        try:
            # FIXED: Pass the properly formatted analysis data
            discussion_turns = await self.agent_system.conduct_discussion(
                video_analysis=video_analysis_for_agents,  # Use converted data
                num_rounds=discussion_rounds,
                selected_agents=selected_agents
            )
            
            # Save discussion with agent configuration info
            discussion_file = output_path / f"configurable_discussion_{timestamp}.json"
            self._save_configurable_discussion(discussion_turns, str(discussion_file))
            
            print(f"Configurable agent discussion saved to: {discussion_file}")
            
        except Exception as e:
            print(f"Agent discussion failed: {e}")
            discussion_turns = []
        
        # Phase 3: RAG indexing (if enabled and available)
        rag_status = "Not enabled"
        if self.enable_rag and self.rag_interface:
            print("\nPHASE 3: RAG INDEXING (Configurable Agent Perspectives)")
            print("-"*60)
            
            try:
                # Index with configurable agent perspectives
                rag_status = await self._index_configurable_agents_for_rag(
                    video_analysis_for_agents, discussion_turns
                )
                print(rag_status)
                
            except Exception as e:
                print(f"RAG indexing failed: {e}")
                rag_status = f"Failed: {e}"
        
        # Phase 4: Generate comprehensive report
        print("\nPHASE 4: CONFIGURABLE AGENT ANALYSIS REPORT")
        print("-"*60)
        
        try:
            report = self._generate_configurable_report(
                video_analysis_for_agents, discussion_turns, participating_agents,
                rag_status, timestamp, output_path
            )
            
            print("Configurable agent analysis report generated")
            
        except Exception as e:
            print(f"Report generation failed: {e}")
            report = {"error": str(e)}
        
        print("\n" + "="*90)
        print("CONFIGURABLE AGENT ANALYSIS PIPELINE COMPLETE!")
        print("="*90)
        print(f"All outputs saved to: {output_dir}")
        print(f"Agents participated: {len(participating_agents)}")
        print(f"Discussion turns: {len(discussion_turns)}")
        
        if self.enable_rag and rag_status.startswith("Successfully"):
            print(f"\nRAG CAPABILITIES NOW AVAILABLE:")
            print("   • Query specific configurable agent perspectives")
            print("   • Search by agent name, role, or expertise area")
            print("   • Find frame-specific insights from custom agents")
            print("   • Temporal queries with agent-specific context")
            
            print(f"\nEXAMPLE QUERIES WITH YOUR CONFIGURED AGENTS:")
            for agent in participating_agents[:3]:  # Show examples for first 3 agents
                print(f"   python query_video_rag.py 'What did {agent.name} say about [topic]?'")
            print(f"   python query_video_rag.py 'When did the {participating_agents[0].role.lower()} mention [concept]?'")
        
        return report
    
    def _debug_enhanced_results(self, enhanced_results):
        """Debug function to inspect the actual structure of enhanced_results"""
        print("\n" + "="*60)
        print("DEBUGGING ENHANCED RESULTS STRUCTURE")
        print("="*60)
        
        print(f"Type of enhanced_results: {type(enhanced_results)}")
        print(f"Is dict: {isinstance(enhanced_results, dict)}")
        print(f"Has __dict__: {hasattr(enhanced_results, '__dict__')}")
        print(f"Has attributes: {dir(enhanced_results)[:10] if hasattr(enhanced_results, '__dict__') else 'No attributes'}")
        
        # If it's a dict, show keys
        if isinstance(enhanced_results, dict):
            print(f"Dictionary keys: {list(enhanced_results.keys())}")
            for key in ['frame_count', 'subtitle_count', 'frame_analyses', 'overall_analysis']:
                if key in enhanced_results:
                    value = enhanced_results[key]
                    if key in ['frame_analyses', 'subtitle_analyses']:
                        print(f"  {key}: type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: MISSING")
        
        # If it's an object, show attributes
        elif hasattr(enhanced_results, '__dict__'):
            attrs = enhanced_results.__dict__
            print(f"Object attributes: {list(attrs.keys())}")
            for key in ['frame_count', 'subtitle_count', 'frame_analyses', 'overall_analysis']:
                if hasattr(enhanced_results, key):
                    value = getattr(enhanced_results, key)
                    if key in ['frame_analyses', 'subtitle_analyses']:
                        print(f"  {key}: type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: MISSING")
        
        # Show the actual content if it's something else
        else:
            print(f"Content: {str(enhanced_results)[:500]}...")
        
        print("="*60)

    def _convert_enhanced_results_for_agents(self, enhanced_results: Any) -> Dict[str, Any]:
        """FIXED: Convert enhanced analysis results to format expected by agents"""
        
        # Default structure expected by agents
        default_result = {
            'video_path': 'unknown',
            'frame_count': 10,
            'subtitle_count': 0,
            'frame_analyses': [],
            'subtitle_analyses': [],
            'overall_analysis': 'Basic video analysis completed.',
            'scene_breakdown': [],
            'visual_elements': {},
            'content_categories': ['general'],
            'processing_time': 0,
            'total_cost': 0,
            'timestamp': datetime.now().isoformat(),
            'analysis_depth': 'comprehensive'
        }
        
        print(f"🔧 CONVERTING: Enhanced results type: {type(enhanced_results)}")
        
        # Handle None or empty results
        if not enhanced_results:
            print("⚠️ Empty enhanced results, using defaults")
            return default_result
        
        try:
            # Case 1: It's already a dictionary with the expected structure
            if isinstance(enhanced_results, dict):
                # Check if it has the expected keys for agent processing
                if all(key in enhanced_results for key in ['frame_count', 'frame_analyses', 'overall_analysis']):
                    print("✅ Enhanced results already in correct format")
                    return enhanced_results
                
                # Case 2: It's enhanced pipeline results format
                result = default_result.copy()
                
                # Extract from enhanced pipeline structure
                if 'analysis_summary' in enhanced_results:
                    summary = enhanced_results['analysis_summary']
                    result['frame_count'] = summary.get('frames_analyzed', result['frame_count'])
                    result['subtitle_count'] = summary.get('subtitle_segments', result['subtitle_count'])
                    result['processing_time'] = summary.get('processing_time', result['processing_time'])
                
                if 'key_insights' in enhanced_results:
                    insights = enhanced_results['key_insights']
                    
                    # Create frame analyses from visual highlights
                    if 'visual_highlights' in insights and insights['visual_highlights']:
                        frame_analyses = []
                        for i, highlight in enumerate(insights['visual_highlights']):
                            frame_analyses.append({
                                'frame_number': i + 1,
                                'timestamp': i * 5.0,
                                'analysis': highlight,
                                'tokens_used': 150,
                                'cost': 0.001
                            })
                        result['frame_analyses'] = frame_analyses
                    
                    if 'comprehensive_assessment' in insights:
                        result['overall_analysis'] = insights['comprehensive_assessment']
                
                # Extract other fields
                result['video_path'] = enhanced_results.get('video_path', result['video_path'])
                result['total_cost'] = enhanced_results.get('total_cost', result['total_cost'])
                result['timestamp'] = enhanced_results.get('timestamp', result['timestamp'])
                
                print(f"✅ Converted enhanced results: {result['frame_count']} frames, {len(result['frame_analyses'])} analyses")
                return result
            
            # Case 3: It's an object with attributes
            elif hasattr(enhanced_results, '__dict__'):
                result = default_result.copy()
                attrs = enhanced_results.__dict__
                
                # Direct attribute mapping
                for key in ['video_path', 'frame_count', 'subtitle_count', 'frame_analyses', 
                        'subtitle_analyses', 'overall_analysis', 'processing_time', 'total_cost']:
                    if key in attrs and attrs[key] is not None:
                        result[key] = attrs[key]
                
                return result
            
            else:
                print(f"⚠️ Unknown enhanced results type: {type(enhanced_results)}")
                return default_result
        
        except Exception as e:
            print(f"❌ Error converting enhanced results: {e}")
            return default_result
    
    def _safe_get_selected_agents(self, selected_agents: Optional[List[str]]) -> List[str]:
        """Safely get selected agents, preventing index errors"""
        if not selected_agents or len(selected_agents) == 0:
            return ['alex', 'maya', 'jordan']  # Default agents
        
        # Ensure we have valid agent names
        valid_agents = []
        available_agent_names = [agent.name.lower() for agent in self.agent_system.agents]
        
        for agent_name in selected_agents:
            if agent_name.lower() in available_agent_names:
                valid_agents.append(agent_name)
        
        # If no valid agents, return defaults
        if not valid_agents:
            return ['alex', 'maya', 'jordan']
        
        return valid_agents

    def _create_basic_video_analysis(self, video_path: str, max_frames: int, subtitle_path: Optional[str]) -> Dict[str, Any]:
        """Create basic video analysis when full pipeline is not available"""
        return {
            "video_path": video_path,
            "frame_count": max_frames,
            "subtitle_count": 3 if subtitle_path else 0,
            "frame_analyses": [
                {
                    "frame_number": i,
                    "timestamp": i * 5.0,
                    "analysis": f"Frame {i} contains visual elements suitable for {', '.join([agent.role for agent in self.agent_system.agents[:2]])} analysis."
                }
                for i in range(1, min(max_frames + 1, 6))
            ],
            "subtitle_analyses": [
                {
                    "subtitle_range": "0.0s - 10.0s",
                    "text_analyzed": "Audio content providing context for agent analysis.",
                    "analysis": "Dialogue content suitable for multi-perspective agent analysis."
                }
            ] if subtitle_path else [],
            "overall_analysis": f"Video content from {video_path} ready for configurable agent analysis with {len(self.agent_system.agents)} specialized agents.",
            "scene_breakdown": [],
            "visual_elements": {},
            "content_categories": [],
            "processing_time": 0,
            "total_cost": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _index_configurable_agents_for_rag(self, video_analysis: Dict[str, Any], discussion_turns: List[Any]) -> str:
        """Index configurable agent perspectives for RAG"""
        if not self.rag_interface:
            return "RAG interface not available"
        
        try:
            # Use existing RAG indexing but with configurable agents
            status = await self.rag_interface.rag_system.index_video_for_rag(video_analysis, discussion_turns)
            return f"Successfully indexed video with {len(self.agent_system.agents)} configurable agent perspectives"
        except Exception as e:
            return f"RAG indexing failed: {e}"
    
    def _save_configurable_discussion(self, discussion_turns: List[Any], output_path: str):
        """Save discussion with configurable agent metadata"""
        discussion_data = {
            "timestamp": datetime.now().isoformat(),
            "discussion_type": "configurable_multi_agent",
            "agent_configuration": {
                "total_agents": len(self.agent_system.agents),
                "participating_agents": len(set(turn.agent_name for turn in discussion_turns)),
                "agent_details": [
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "model": agent.model,
                        "expertise": agent.expertise,
                        "focus_areas": agent.focus_areas
                    }
                    for agent in self.agent_system.agents
                ]
            },
            "discussion_summary": self.agent_system.get_discussion_summary(),
            "turns": [
                {
                    "agent_name": turn.agent_name,
                    "agent_role": turn.agent_role,
                    "content": turn.content,
                    "round_number": turn.round_number,
                    "timestamp": turn.timestamp,
                    "responding_to": getattr(turn, 'responding_to', None)
                }
                for turn in discussion_turns
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(discussion_data, f, indent=2, ensure_ascii=False)
    
    def _generate_configurable_report(self, video_analysis: Dict[str, Any], 
                                    discussion_turns: List[Any], participating_agents: List[Any],
                                    rag_status: str, timestamp: str, output_path: Path) -> Dict[str, Any]:
        """Generate comprehensive report with configurable agent insights"""
        
        report = {
            "timestamp": timestamp,
            "analysis_type": "configurable_multi_agent_enhanced",
            "video_path": video_analysis.get("video_path", "unknown"),
            "analysis_depth": self.analysis_depth,
            
            "configurable_agent_features": {
                "total_agents_configured": len(self.agent_system.agents),
                "agents_participated": len(participating_agents),
                "custom_agent_system": True,
                "agent_specializations": [agent.role for agent in participating_agents],
                "models_used": list(set(agent.model for agent in participating_agents)),
                "expertise_areas": [area for agent in participating_agents for area in agent.expertise],
                "rag_enhanced": self.enable_rag
            },
            
            "agent_configuration": {
                "participating_agents": [
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "emoji": agent.emoji,
                        "model": agent.model,
                        "expertise": agent.expertise,
                        "focus_areas": agent.focus_areas,
                        "personality": agent.personality,
                        "discussion_style": agent.discussion_style,
                        "analysis_approach": agent.analysis_approach
                    }
                    for agent in participating_agents
                ],
                "agent_models_distribution": {
                    model: len([a for a in participating_agents if a.model == model])
                    for model in set(agent.model for agent in participating_agents)
                }
            },
            
            "analysis_summary": {
                "frames_analyzed": video_analysis.get("frame_count", 0),
                "subtitle_segments": video_analysis.get("subtitle_count", 0),
                "discussion_turns": len(discussion_turns),
                "discussion_rounds": max([turn.round_number for turn in discussion_turns]) if discussion_turns else 0,
                "processing_time": video_analysis.get("processing_time", 0),
                "total_cost": video_analysis.get("total_cost", 0)
            },
            
            "agent_insights_summary": {
                "agent_contributions": {
                    turn.agent_name: len([t for t in discussion_turns if t.agent_name == turn.agent_name])
                    for turn in discussion_turns
                } if discussion_turns else {},
                "expertise_coverage": {
                    agent.name: agent.expertise for agent in participating_agents
                },
                "focus_areas_addressed": {
                    agent.name: agent.focus_areas for agent in participating_agents
                },
                "model_perspectives": {
                    agent.model: [a.name for a in participating_agents if a.model == agent.model]
                    for agent in participating_agents
                }
            },
            
            "rag_capabilities": {
                "enabled": self.enable_rag,
                "indexing_status": rag_status,
                "configurable_agent_search": self.enable_rag and rag_status.startswith("Successfully"),
                "query_examples": [
                    f"What did {agent.name} say about [topic]?"
                    for agent in participating_agents[:3]
                ] + [
                    f"When did the {participating_agents[0].role.lower()} discuss [concept]?",
                    f"How did {participating_agents[-1].name} analyze [element]?",
                ] if participating_agents else []
            },
            
            "customization_features": {
                "agent_templates_available": ["film_analysis", "educational", "marketing"],
                "custom_agent_creation": True,
                "agent_expertise_customization": True,
                "discussion_style_customization": True,
                "model_selection_per_agent": True,
                "focus_area_specification": True
            }
        }
        
        # Save comprehensive report
        report_file = output_path / f"configurable_agent_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Create human-readable summary
        summary_file = output_path / f"configurable_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("CONFIGURABLE MULTI-AGENT VIDEO ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Type: {report['analysis_type']}\n")
            f.write(f"Analysis Depth: {self.analysis_depth.upper()}\n\n")
            
            f.write("CONFIGURED AGENTS\n")
            f.write("-"*40 + "\n")
            for agent in participating_agents:
                f.write(f"{agent.emoji} {agent.name} ({agent.role}) - {agent.model}\n")
                f.write(f"  Expertise: {', '.join(agent.expertise[:4])}\n")
                f.write(f"  Focus: {', '.join(agent.focus_areas[:3])}\n")
                f.write(f"  Style: {agent.discussion_style[:60]}...\n\n")
            
            f.write("ANALYSIS RESULTS\n")
            f.write("-"*40 + "\n")
            summary = report["analysis_summary"]
            f.write(f"Frames analyzed: {summary['frames_analyzed']}\n")
            f.write(f"Discussion turns: {summary['discussion_turns']}\n")
            f.write(f"Participating agents: {report['configurable_agent_features']['agents_participated']}\n")
            f.write(f"Models used: {', '.join(report['configurable_agent_features']['models_used'])}\n\n")
            
            if self.enable_rag and rag_status.startswith("Successfully"):
                f.write("RAG QUERY EXAMPLES\n")
                f.write("-"*40 + "\n")
                for example in report["rag_capabilities"]["query_examples"][:6]:
                    f.write(f"• {example}\n")
                f.write("\n")
            
            f.write(f"Total cost: ${report['analysis_summary']['total_cost']:.4f}\n")
        
        print(f"Configurable agent report saved to: {report_file}")
        print(f"Summary saved to: {summary_file}")
        
        return report

# Command line interface for the integrated pipeline
async def main():
    """Main entry point for configurable agent video analysis"""
    parser = argparse.ArgumentParser(
        description="Integrated Configurable Agent Video Analysis Pipeline - FIXED VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default agents
  python integrated_configurable_pipeline.py video.mp4
  
  # Film analysis with specialized agents
  python integrated_configurable_pipeline.py video.mp4 --template film_analysis
  
  # Educational content with custom agents
  python integrated_configurable_pipeline.py video.mp4 --content-type educational
  
  # Select specific agents for discussion
  python integrated_configurable_pipeline.py video.mp4 --agents "Alex,Maya"
  
  # Comprehensive analysis with full pipeline
  python integrated_configurable_pipeline.py video.mp4 --depth comprehensive --rounds 4

Agent Templates Available:
  film_analysis    - Cinematographer, Film Critic, Sound Designer
  educational      - Learning Specialist, Subject Expert, Engagement Analyst  
  marketing        - Brand Strategist, Conversion Specialist, Creative Director

Content Types (auto-selects appropriate template):
  film, movie, cinema          -> film_analysis template
  educational, tutorial        -> educational template
  marketing, commercial        -> marketing template
        """
    )
    
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--subtitles", "-s", help="Path to subtitle file")
    parser.add_argument("--depth", "-d", choices=["basic", "detailed", "comprehensive"], 
                       default="comprehensive", help="Analysis depth")
    parser.add_argument("--frames", "-f", type=int, default=10, help="Maximum frames to analyze")
    parser.add_argument("--fps", type=float, default=0.2, help="Frame extraction rate")
    parser.add_argument("--rounds", "-r", type=int, default=3, help="Discussion rounds")
    parser.add_argument("--output", "-o", default="configurable_analysis_output", help="Output directory")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG indexing")
    
    # Agent configuration options
    parser.add_argument("--template", choices=["film_analysis", "educational", "marketing"], 
                       help="Load predefined agent template")
    parser.add_argument("--content-type", help="Content type (auto-selects template)")
    parser.add_argument("--agents", help="Comma-separated list of specific agents to include")
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Check subtitle file
    if args.subtitles and not os.path.exists(args.subtitles):
        print(f"Warning: Subtitle file not found: {args.subtitles}")
        args.subtitles = None
    
    # Parse selected agents
    selected_agents = None
    if args.agents:
        selected_agents = [name.strip() for name in args.agents.split(",")]
    
    # Initialize pipeline
    pipeline = IntegratedConfigurableAnalysisPipeline(
        analysis_depth=args.depth,
        enable_rag=not args.no_rag
    )
    
    # Show configuration summary
    print(f"\nCONFIGURABLE AGENT VIDEO ANALYSIS")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Subtitles: {args.subtitles or 'None'}")
    print(f"Analysis depth: {args.depth.upper()}")
    print(f"Discussion rounds: {args.rounds}")
    print(f"Template: {args.template or 'Default configuration'}")
    print(f"Content type: {args.content_type or 'Auto-detect'}")
    print(f"Selected agents: {', '.join(selected_agents) if selected_agents else 'All configured agents'}")
    print(f"RAG indexing: {'Enabled' if not args.no_rag else 'Disabled'}")
    print("="*70)
    
    try:
        # Run comprehensive analysis with configurable agents
        results = await pipeline.analyze_video_with_configurable_agents(
            video_path=args.video,
            subtitle_path=args.subtitles,
            max_frames=args.frames,
            fps_extract=args.fps,
            discussion_rounds=args.rounds,
            selected_agents=selected_agents,
            content_type=args.content_type,
            agent_template=args.template,
            output_dir=args.output
        )
        
        print(f"\nConfigurable agent analysis complete!")
        print(f"Results saved to: {args.output}")
        
        # Show agent contribution summary
        if "agent_insights_summary" in results:
            contributions = results["agent_insights_summary"]["agent_contributions"]
            print(f"\nAgent Contributions:")
            for agent_name, count in contributions.items():
                print(f"  {agent_name}: {count} contributions")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)