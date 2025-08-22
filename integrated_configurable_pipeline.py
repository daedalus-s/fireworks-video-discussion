#!/usr/bin/env python3
"""
Integrated Configurable Agent Pipeline
Combines your existing video analysis with configurable multi-agent system
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
    from integrated_rag_pipeline import ComprehensiveRAGPipeline
    from rag_enhanced_vector_system import RAGQueryInterface
    from enhanced_descriptive_analysis import analyze_video_highly_descriptive
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Analysis modules not fully available: {e}")
    ANALYSIS_MODULES_AVAILABLE = False

class IntegratedConfigurableAnalysisPipeline:
    """Enhanced pipeline with configurable agents and full video analysis"""
    
    def __init__(self, 
                 analysis_depth: str = "comprehensive",
                 enable_rag: bool = True):
        """Initialize the integrated pipeline"""
        self.analysis_depth = analysis_depth
        self.enable_rag = enable_rag
        
        # Initialize configurable agent system
        self.agent_system = ConfigurableMultiAgentDiscussion()
        
        # Initialize existing analysis systems if available
        if ANALYSIS_MODULES_AVAILABLE:
            try:
                self.rag_pipeline = ComprehensiveRAGPipeline(
                    analysis_depth=analysis_depth,
                    enable_rag=enable_rag
                )
                self.rag_interface = RAGQueryInterface() if enable_rag else None
                print("✅ Full analysis pipeline initialized")
            except Exception as e:
                print(f"⚠️ RAG pipeline initialization failed: {e}")
                self.rag_pipeline = None
                self.rag_interface = None
        else:
            self.rag_pipeline = None
            self.rag_interface = None
        
        print("✅ Integrated Configurable Analysis Pipeline initialized")
    
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
            "technical": "technical_docs",
            "documentation": "technical_docs",
            "default": None
        }
        
        template_name = templates.get(content_type.lower(), "default")
        
        if template_name:
            template_agents = load_agent_template(template_name)
            if template_agents:
                self.agent_system.agents = template_agents
                print(f"✅ Loaded {len(template_agents)} agents for {content_type} content")
                return True
        
        # Keep default agents if no template found
        print(f"ℹ️ Using default agents for {content_type} content")
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
        Complete analysis with configurable agents
        
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
        print("🎬 INTEGRATED CONFIGURABLE AGENT VIDEO ANALYSIS PIPELINE")
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
                print(f"✅ Loaded agent template: {agent_template}")
        elif content_type:
            self.setup_agents_for_content_type(content_type)
        
        # Show participating agents
        participating_agents = self.agent_system.agents
        if selected_agents:
            participating_agents = [a for a in self.agent_system.agents if a.name in selected_agents]
        
        print(f"\n🤖 CONFIGURED AGENTS ({len(participating_agents)}):")
        print("-"*60)
        for agent in participating_agents:
            print(f"  {agent.emoji} {agent.name} ({agent.role}) - Model: {agent.model}")
            print(f"    Expertise: {', '.join(agent.expertise[:3])}")
            print(f"    Focus: {', '.join(agent.focus_areas[:2])}")
        
        # Phase 1: Enhanced descriptive analysis (if available)
        video_analysis_results = None
        if ANALYSIS_MODULES_AVAILABLE and self.rag_pipeline:
            print("\n📊 PHASE 1: ENHANCED DESCRIPTIVE ANALYSIS")
            print("-"*60)
            
            try:
                video_analysis_results = await analyze_video_highly_descriptive(
                    video_path=video_path,
                    subtitle_path=subtitle_path,
                    max_frames=max_frames,
                    fps_extract=fps_extract,
                    analysis_depth=self.analysis_depth,
                    enable_vector_search=True,
                    output_dir=output_dir
                )
                print("✅ Enhanced descriptive analysis complete")
                
            except Exception as e:
                print(f"⚠️ Enhanced analysis failed, using basic analysis: {e}")
                video_analysis_results = self._create_basic_video_analysis(
                    video_path, max_frames, subtitle_path
                )
        else:
            print("\n📊 PHASE 1: BASIC VIDEO ANALYSIS")
            print("-"*60)
            video_analysis_results = self._create_basic_video_analysis(
                video_path, max_frames, subtitle_path
            )
        
        # Phase 2: Configurable multi-agent discussion
        print("\n💬 PHASE 2: CONFIGURABLE MULTI-AGENT DISCUSSION")
        print("-"*60)
        
        try:
            discussion_turns = await self.agent_system.conduct_discussion(
                video_analysis=video_analysis_results,
                num_rounds=discussion_rounds,
                selected_agents=selected_agents
            )
            
            # Save discussion with agent configuration info
            discussion_file = output_path / f"configurable_discussion_{timestamp}.json"
            self._save_configurable_discussion(discussion_turns, str(discussion_file))
            
            print(f"✅ Configurable agent discussion saved to: {discussion_file}")
            
        except Exception as e:
            print(f"❌ Agent discussion failed: {e}")
            discussion_turns = []
        
        # Phase 3: RAG indexing (if enabled and available)
        rag_status = "Not enabled"
        if self.enable_rag and ANALYSIS_MODULES_AVAILABLE and self.rag_interface:
            print("\n🧠 PHASE 3: RAG INDEXING (Configurable Agent Perspectives)")
            print("-"*60)
            
            try:
                # Index with configurable agent perspectives
                rag_status = await self._index_configurable_agents_for_rag(
                    video_analysis_results, discussion_turns
                )
                print(rag_status)
                
            except Exception as e:
                print(f"❌ RAG indexing failed: {e}")
                rag_status = f"Failed: {e}"
        
        # Phase 4: Generate comprehensive report
        print("\n📄 PHASE 4: CONFIGURABLE AGENT ANALYSIS REPORT")
        print("-"*60)
        
        try:
            report = self._generate_configurable_report(
                video_analysis_results, discussion_turns, participating_agents,
                rag_status, timestamp, output_path
            )
            
            print("✅ Configurable agent analysis report generated")
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            report = {"error": str(e)}
        
        print("\n" + "="*90)
        print("✅ CONFIGURABLE AGENT ANALYSIS PIPELINE COMPLETE!")
        print("="*90)
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"🤖 Agents participated: {len(participating_agents)}")
        print(f"💬 Discussion turns: {len(discussion_turns)}")
        
        if self.enable_rag and rag_status.startswith("✅"):
            print(f"\n🧠 RAG CAPABILITIES NOW AVAILABLE:")
            print("   • Query specific configurable agent perspectives")
            print("   • Search by agent name, role, or expertise area")
            print("   • Find frame-specific insights from custom agents")
            print("   • Temporal queries with agent-specific context")
            
            print(f"\n🔍 EXAMPLE QUERIES WITH YOUR CONFIGURED AGENTS:")
            for agent in participating_agents[:3]:  # Show examples for first 3 agents
                print(f"   python query_video_rag.py 'What did {agent.name} say about [topic]?'")
            print(f"   python query_video_rag.py 'When did the {participating_agents[0].role.lower()} mention [concept]?'")
        
        return report
    
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
            "overall_analysis": f"Video content from {video_path} ready for configurable agent analysis with {len(self.agent_system.agents)} specialized agents."
        }
    
    async def _index_configurable_agents_for_rag(self, video_analysis: Dict[str, Any], discussion_turns: List) -> str:
        """Index configurable agent perspectives for RAG"""
        if not self.rag_interface:
            return "❌ RAG interface not available"
        
        try:
            # Use existing RAG indexing but with configurable agents
            status = await self.rag_interface.rag_system.index_video_for_rag(video_analysis, discussion_turns)
            return f"✅ Successfully indexed video with {len(self.agent_system.agents)} configurable agent perspectives"
        except Exception as e:
            return f"❌ RAG indexing failed: {e}"
    
    def _save_configurable_discussion(self, discussion_turns: List, output_path: str):
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
                    "responding_to": turn.responding_to
                }
                for turn in discussion_turns
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(discussion_data, f, indent=2, ensure_ascii=False)
    
    def _generate_configurable_report(self, video_analysis: Dict[str, Any], 
                                    discussion_turns: List, participating_agents: List,
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
                "processing_time": "N/A",  # Could be calculated
                "total_cost": video_analysis.get("total_cost", 0)
            },
            
            "agent_insights_summary": {
                "agent_contributions": {
                    turn.agent_name: len([t for t in discussion_turns if t.agent_name == turn.agent_name])
                    for turn in discussion_turns
                },
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
                "configurable_agent_search": self.enable_rag and rag_status.startswith("✅"),
                "query_examples": [
                    f"What did {agent.name} say about [topic]?"
                    for agent in participating_agents[:3]
                ] + [
                    f"When did the {participating_agents[0].role.lower()} discuss [concept]?",
                    f"How did {participating_agents[-1].name} analyze [element]?",
                    f"At what frame did agents with {participating_agents[0].expertise[0]} expertise comment?"
                ] if participating_agents else []
            },
            
            "customization_features": {
                "agent_templates_available": ["film_analysis", "educational", "marketing", "technical_docs"],
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
            
            f.write("CONFIGURABLE FEATURES\n")
            f.write("-"*40 + "\n")
            f.write("✅ Custom agent personalities and roles\n")
            f.write("✅ Flexible expertise areas and focus\n")
            f.write("✅ Multiple AI models per discussion\n")
            f.write("✅ Agent template system\n")
            f.write("✅ Interactive agent configuration\n")
            if self.enable_rag:
                f.write("✅ RAG-enhanced agent perspective search\n")
            f.write("\n")
            
            f.write("ANALYSIS RESULTS\n")
            f.write("-"*40 + "\n")
            summary = report["analysis_summary"]
            f.write(f"Frames analyzed: {summary['frames_analyzed']}\n")
            f.write(f"Discussion turns: {summary['discussion_turns']}\n")
            f.write(f"Participating agents: {report['configurable_agent_features']['agents_participated']}\n")
            f.write(f"Models used: {', '.join(report['configurable_agent_features']['models_used'])}\n\n")
            
            if self.enable_rag and rag_status.startswith("✅"):
                f.write("RAG QUERY EXAMPLES\n")
                f.write("-"*40 + "\n")
                for example in report["rag_capabilities"]["query_examples"][:6]:
                    f.write(f"• {example}\n")
                f.write("\n")
            
            f.write(f"Total cost: ${report['analysis_summary']['total_cost']:.4f}\n")
        
        print(f"✅ Configurable agent report saved to: {report_file}")
        print(f"✅ Summary saved to: {summary_file}")
        
        return report

# Command line interface for the integrated pipeline
async def main():
    """Main entry point for configurable agent video analysis"""
    parser = argparse.ArgumentParser(
        description="Integrated Configurable Agent Video Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default agents
  python integrated_configurable_pipeline.py video.mp4
  
  # Film analysis with specialized agents
  python integrated_configurable_pipeline.py video.mp4 --template film_analysis
  
  # Educational content with custom agents
  python integrated_configurable_pipeline.py video.mp4 --content-type educational
  
  # Configure agents interactively then analyze
  python integrated_configurable_pipeline.py video.mp4 --configure-agents
  
  # Select specific agents for discussion
  python integrated_configurable_pipeline.py video.mp4 --agents "Alex,Maya"
  
  # Comprehensive analysis with full pipeline
  python integrated_configurable_pipeline.py video.mp4 --depth comprehensive --rounds 4

Agent Templates Available:
  film_analysis    - Cinematographer, Film Critic, Sound Designer
  educational      - Learning Specialist, Subject Expert, Engagement Analyst  
  marketing        - Brand Strategist, Conversion Specialist, Creative Director
  technical_docs   - Technical Writer, UX Researcher

Content Types (auto-selects appropriate template):
  film, movie, cinema          -> film_analysis template
  educational, tutorial        -> educational template
  marketing, commercial        -> marketing template
  technical, documentation     -> technical_docs template
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
    parser.add_argument("--configure-agents", action="store_true", help="Configure agents interactively before analysis")
    parser.add_argument("--template", choices=["film_analysis", "educational", "marketing", "technical_docs"], 
                       help="Load predefined agent template")
    parser.add_argument("--content-type", help="Content type (auto-selects template)")
    parser.add_argument("--agents", help="Comma-separated list of specific agents to include")
    parser.add_argument("--list-agents", action="store_true", help="List configured agents and exit")
    parser.add_argument("--list-templates", action="store_true", help="List available templates and exit")
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_templates:
        show_available_templates()
        return 0
    
    # Initialize pipeline
    pipeline = IntegratedConfigurableAnalysisPipeline(
        analysis_depth=args.depth,
        enable_rag=not args.no_rag
    )
    
    if args.list_agents:
        print(f"\nConfigured agents ({len(pipeline.agent_system.agents)}):")
        for agent in pipeline.agent_system.agents:
            print(f"  {agent.emoji} {agent.name} ({agent.role}) - {agent.model}")
            print(f"    Expertise: {', '.join(agent.expertise[:4])}")
        return 0
    
    # Validate video file
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return 1
    
    # Check subtitle file
    if args.subtitles and not os.path.exists(args.subtitles):
        print(f"⚠️ Warning: Subtitle file not found: {args.subtitles}")
        args.subtitles = None
    
    # Interactive agent configuration
    if args.configure_agents:
        print("🤖 Interactive Agent Configuration")
        pipeline.agent_system.configure_agents_interactive()
    
    # Parse selected agents
    selected_agents = None
    if args.agents:
        selected_agents = [name.strip() for name in args.agents.split(",")]
        # Validate agent names
        available_names = [agent.name for agent in pipeline.agent_system.agents]
        invalid_agents = [name for name in selected_agents if name not in available_names]
        if invalid_agents:
            print(f"❌ Invalid agent names: {invalid_agents}")
            print(f"Available agents: {', '.join(available_names)}")
            return 1
    
    # Show configuration summary
    print(f"\n🎬 CONFIGURABLE AGENT VIDEO ANALYSIS")
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
        
        print(f"\n🎉 Configurable agent analysis complete!")
        print(f"📁 Results saved to: {args.output}")
        
        # Show agent contribution summary
        if "agent_insights_summary" in results:
            contributions = results["agent_insights_summary"]["agent_contributions"]
            print(f"\n🤖 Agent Contributions:")
            for agent_name, count in contributions.items():
                agent = pipeline.agent_system.get_agent(agent_name)
                emoji = agent.emoji if agent else "🤖"
                print(f"  {emoji} {agent_name}: {count} contributions")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

# Quick setup utilities
def quick_setup_for_content_type(content_type: str):
    """Quick setup utility for specific content types"""
    pipeline = IntegratedConfigurableAnalysisPipeline()
    
    print(f"🚀 Quick Setup for {content_type.title()} Content")
    print("="*50)
    
    success = pipeline.setup_agents_for_content_type(content_type)
    
    if success:
        print(f"\n✅ Agents configured for {content_type} analysis:")
        for agent in pipeline.agent_system.agents:
            print(f"  {agent.emoji} {agent.name} ({agent.role})")
            print(f"    Expertise: {', '.join(agent.expertise[:3])}")
    
    # Save configuration
    config_file = f"{content_type}_agent_config.json"
    pipeline.agent_system.export_agent_configuration(config_file)
    print(f"\n💾 Configuration saved to: {config_file}")
    
    return pipeline

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ["setup-film", "setup-educational", "setup-marketing", "setup-technical"]:
        content_type = sys.argv[1].replace("setup-", "")
        quick_setup_for_content_type(content_type)
    else:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)