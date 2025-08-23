"""
Complete Configurable Multi-Agent Discussion System
Includes all missing functions and agent templates
"""

import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random

from fireworks_client import FireworksClient
from simple_api_manager import SimpleAPIManager  # or OptimizedAPIManager for speed
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CustomAgent:
    """Represents a custom AI agent with user-defined characteristics"""
    name: str
    role: str
    personality: str
    expertise: List[str]
    discussion_style: str
    model: str  # Which Fireworks model to use
    emoji: str
    focus_areas: List[str]
    analysis_approach: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomAgent':
        """Create agent from dictionary"""
        return cls(**data)

@dataclass
class AgentDiscussionTurn:
    """A single turn in the agent discussion"""
    agent_name: str
    agent_role: str
    content: str
    timestamp: str
    round_number: int
    responding_to: Optional[str] = None
    frame_references: List[Dict[str, Any]] = None
    timestamp_references: List[Dict[str, Any]] = None

class AgentTemplates:
    """Predefined agent templates for different content types"""
    
    @staticmethod
    def get_film_analysis_agents() -> List[CustomAgent]:
        """Film analysis specialist agents"""
        return [
            CustomAgent(
                name="Cinematographer Chen",
                role="Director of Photography",
                personality="Visually focused and technically precise. Analyzes camera work, lighting, and visual composition with professional expertise.",
                expertise=["cinematography", "camera movements", "lighting design", "visual composition", "color grading", "lens selection"],
                discussion_style="Technical and precise, references specific cinematographic techniques and equipment",
                model="vision",  # Best for visual analysis
                emoji="🎥",
                focus_areas=["camera work", "lighting", "visual composition", "cinematographic techniques"],
                analysis_approach="Technical analysis of visual elements, camera techniques, and photographic principles"
            ),
            CustomAgent(
                name="Isabella Film Critic",
                role="Film Studies Scholar",
                personality="Analytical and culturally aware. Evaluates films within broader cinematic contexts and artistic traditions.",
                expertise=["film theory", "cinematic history", "genre analysis", "narrative structure", "cultural critique", "artistic merit"],
                discussion_style="Scholarly and contextual, references film history and theoretical frameworks",
                model="qwen3",  # Great for creative analysis
                emoji="🎭",
                focus_areas=["narrative structure", "artistic merit", "cinematic tradition", "cultural context"],
                analysis_approach="Academic and theoretical, placing content within broader cinematic and cultural contexts"
            ),
            CustomAgent(
                name="Marcus Sound Designer",
                role="Audio Post-Production Specialist",
                personality="Aurally focused and detail-oriented. Analyzes sound design, music, and audio storytelling elements.",
                expertise=["sound design", "musical score", "dialogue editing", "ambient sound", "audio mixing", "acoustic storytelling"],
                discussion_style="Audio-focused and technical, discusses sound's role in storytelling",
                model="gpt_oss",  # Balanced analysis
                emoji="🎵",
                focus_areas=["sound design", "musical elements", "audio storytelling", "acoustic atmosphere"],
                analysis_approach="Audio-centric analysis focusing on sound's contribution to narrative and emotional impact"
            )
        ]
    
    @staticmethod
    def get_educational_agents() -> List[CustomAgent]:
        """Educational content specialist agents"""
        return [
            CustomAgent(
                name="Dr. Learning Specialist",
                role="Educational Psychology Expert",
                personality="Student-focused and pedagogically minded. Evaluates content for learning effectiveness and engagement.",
                expertise=["learning theory", "cognitive psychology", "educational design", "student engagement", "knowledge retention", "instructional methods"],
                discussion_style="Research-based and learner-centered, focuses on educational outcomes",
                model="qwen3",
                emoji="🧠",
                focus_areas=["learning effectiveness", "student engagement", "knowledge retention", "cognitive load"],
                analysis_approach="Evidence-based educational analysis focusing on learning outcomes and student experience"
            ),
            CustomAgent(
                name="Prof. Subject Expert",
                role="Domain Knowledge Specialist",
                personality="Content-focused and accuracy-driven. Ensures subject matter accuracy and depth.",
                expertise=["subject matter expertise", "content accuracy", "factual verification", "domain knowledge", "academic rigor", "information quality"],
                discussion_style="Authoritative and precise, emphasizes accuracy and completeness",
                model="gpt_oss",
                emoji="📚",
                focus_areas=["content accuracy", "subject depth", "factual correctness", "academic quality"],
                analysis_approach="Subject matter validation with emphasis on accuracy, completeness, and academic standards"
            ),
            CustomAgent(
                name="Sam Engagement Analyst",
                role="Learning Engagement Specialist",
                personality="Motivation-focused and practical. Analyzes content for student motivation and sustained attention.",
                expertise=["engagement strategies", "motivation theory", "attention management", "interactive design", "student psychology", "retention techniques"],
                discussion_style="Practical and student-focused, emphasizes real-world classroom application",
                model="vision",
                emoji="🎯",
                focus_areas=["student motivation", "engagement techniques", "attention retention", "interactive elements"],
                analysis_approach="Practical analysis of engagement factors and motivational elements in educational content"
            )
        ]
    
    @staticmethod
    def get_marketing_agents() -> List[CustomAgent]:
        """Marketing content specialist agents"""
        return [
            CustomAgent(
                name="Alex Brand Strategist",
                role="Brand Strategy Director",
                personality="Brand-focused and strategic. Analyzes content for brand alignment and strategic messaging.",
                expertise=["brand strategy", "messaging consistency", "brand identity", "market positioning", "brand storytelling", "strategic communication"],
                discussion_style="Strategic and brand-focused, evaluates alignment with brand objectives",
                model="qwen3",
                emoji="🏢",
                focus_areas=["brand consistency", "strategic messaging", "brand identity", "market positioning"],
                analysis_approach="Strategic brand analysis focusing on message alignment and brand equity building"
            ),
            CustomAgent(
                name="Casey Conversion Specialist",
                role="Performance Marketing Expert",
                personality="Results-driven and data-focused. Evaluates content for conversion potential and performance metrics.",
                expertise=["conversion optimization", "performance metrics", "user journey", "call-to-action design", "funnel analysis", "behavioral psychology"],
                discussion_style="Data-driven and results-oriented, focuses on measurable outcomes",
                model="gpt_oss",
                emoji="📈",
                focus_areas=["conversion potential", "user journey", "performance metrics", "behavioral triggers"],
                analysis_approach="Performance-focused analysis emphasizing conversion optimization and measurable business outcomes"
            ),
            CustomAgent(
                name="Maya Creative Director",
                role="Creative Strategy Lead",
                personality="Creatively driven and audience-focused. Evaluates creative execution and emotional impact.",
                expertise=["creative strategy", "visual communication", "emotional resonance", "audience psychology", "creative execution", "artistic direction"],
                discussion_style="Creative and emotionally intelligent, focuses on artistic and emotional impact",
                model="vision",
                emoji="🎨",
                focus_areas=["creative execution", "emotional impact", "visual appeal", "audience connection"],
                analysis_approach="Creative analysis focusing on artistic merit, emotional resonance, and audience engagement"
            )
        ]
    
    @staticmethod
    def get_technical_docs_agents() -> List[CustomAgent]:
        """Technical documentation specialist agents"""
        return [
            CustomAgent(
                name="Taylor Technical Writer",
                role="Technical Documentation Specialist",
                personality="Clarity-focused and user-centered. Evaluates content for technical accuracy and usability.",
                expertise=["technical writing", "documentation standards", "user experience", "information architecture", "clarity optimization", "technical communication"],
                discussion_style="Clear and structured, emphasizes usability and comprehension",
                model="gpt_oss",
                emoji="📝",
                focus_areas=["technical clarity", "user experience", "documentation quality", "information structure"],
                analysis_approach="Technical communication analysis focusing on clarity, usability, and effective knowledge transfer"
            ),
            CustomAgent(
                name="Jordan UX Researcher",
                role="User Experience Research Lead",
                personality="User-focused and research-driven. Analyzes content from the user's perspective and interaction patterns.",
                expertise=["user research", "usability testing", "interaction design", "user psychology", "accessibility", "user journey mapping"],
                discussion_style="Research-based and user-centered, emphasizes empirical user feedback",
                model="vision",
                emoji="👤",
                focus_areas=["user experience", "usability", "accessibility", "user interaction"],
                analysis_approach="User-centered research analysis focusing on usability, accessibility, and user satisfaction"
            ),
            CustomAgent(
                name="Dr. Dev Documentation",
                role="Developer Experience Specialist",
                personality="Developer-focused and implementation-oriented. Evaluates content for developer usability and technical implementation.",
                expertise=["developer experience", "API documentation", "code examples", "implementation guidance", "technical accuracy", "developer workflow"],
                discussion_style="Implementation-focused and technically precise, considers developer workflow",
                model="qwen3",
                emoji="💻",
                focus_areas=["developer experience", "technical implementation", "code quality", "workflow efficiency"],
                analysis_approach="Developer-centric analysis focusing on implementation clarity and technical workflow optimization"
            )
        ]

# Template registry
AGENT_TEMPLATES = {
    "film_analysis": AgentTemplates.get_film_analysis_agents,
    "educational": AgentTemplates.get_educational_agents,
    "marketing": AgentTemplates.get_marketing_agents,
    "technical_docs": AgentTemplates.get_technical_docs_agents
}

def load_agent_template(template_name: str) -> Optional[List[CustomAgent]]:
    """Load a predefined agent template"""
    if template_name in AGENT_TEMPLATES:
        try:
            agents = AGENT_TEMPLATES[template_name]()
            logger.info(f"✅ Loaded {template_name} template with {len(agents)} agents")
            return agents
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            return None
    else:
        logger.warning(f"Template {template_name} not found")
        return None

def show_available_templates():
    """Display available agent templates"""
    print("\n📋 AVAILABLE AGENT TEMPLATES")
    print("="*50)
    
    template_descriptions = {
        "film_analysis": {
            "description": "Film & Cinema Analysis Specialists",
            "agents": ["Cinematographer Chen", "Isabella Film Critic", "Marcus Sound Designer"],
            "use_case": "Movie reviews, film analysis, cinematic content"
        },
        "educational": {
            "description": "Educational Content Specialists", 
            "agents": ["Dr. Learning Specialist", "Prof. Subject Expert", "Sam Engagement Analyst"],
            "use_case": "Educational videos, tutorials, learning content"
        },
        "marketing": {
            "description": "Marketing & Brand Specialists",
            "agents": ["Alex Brand Strategist", "Casey Conversion Specialist", "Maya Creative Director"],
            "use_case": "Marketing videos, commercials, promotional content"
        },
        "technical_docs": {
            "description": "Technical Documentation Specialists",
            "agents": ["Taylor Technical Writer", "Jordan UX Researcher", "Dr. Dev Documentation"],
            "use_case": "Technical tutorials, documentation, instructional content"
        }
    }
    
    for template_name, info in template_descriptions.items():
        print(f"\n🎯 {template_name}")
        print(f"   Description: {info['description']}")
        print(f"   Agents: {', '.join(info['agents'])}")
        print(f"   Best for: {info['use_case']}")
        
        # Show agent details
        agents = load_agent_template(template_name)
        if agents:
            for agent in agents:
                print(f"     {agent.emoji} {agent.name} ({agent.role}) - {agent.model}")
    
    print(f"\n💡 Usage: --template [template_name]")
    print(f"   Example: python script.py video.mp4 --template film_analysis")

class ConfigurableMultiAgentDiscussion:
    """Multi-agent discussion system with configurable agents"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the configurable multi-agent system"""
        self.client = FireworksClient(api_key)
        self.agents: List[CustomAgent] = []
        self.discussion_history: List[AgentDiscussionTurn] = []
        self.agent_configs_file = "agent_configurations.json"
        from simple_api_manager import SimpleAPIManager
        # Load saved agent configurations
        self._load_agent_configurations()
        
        logger.info("✅ Configurable Multi-Agent Discussion System initialized")
    
    def create_default_agents(self) -> List[CustomAgent]:
        """Create default agent configurations"""
        default_agents = [
            CustomAgent(
                name="Alex",
                role="Technical Analyst",
                personality="Detail-oriented and analytical. Focuses on technical aspects with precision and expertise.",
                expertise=["cinematography", "camera techniques", "lighting", "editing", "visual effects", "technical production"],
                discussion_style="Precise and factual, supports arguments with specific technical observations",
                model="gpt_oss",
                emoji="🎬",
                focus_areas=["technical quality", "production values", "camera work", "visual composition"],
                analysis_approach="Analytical and methodical, examining technical execution and craftsmanship"
            ),
            CustomAgent(
                name="Maya",
                role="Creative Interpreter",
                personality="Imaginative and insightful. Explores artistic meanings, themes, and emotional impact.",
                expertise=["storytelling", "symbolism", "themes", "emotional impact", "cultural context", "artistic interpretation"],
                discussion_style="Thoughtful and empathetic, asks deep questions and explores creative meanings",
                model="qwen3",
                emoji="🎨",
                focus_areas=["artistic expression", "narrative themes", "emotional resonance", "symbolic content"],
                analysis_approach="Interpretive and contextual, seeking deeper meanings and artistic significance"
            ),
            CustomAgent(
                name="Jordan",
                role="Audience Advocate",
                personality="Practical and viewer-focused. Considers accessibility, engagement, and audience experience.",
                expertise=["user experience", "audience engagement", "accessibility", "clarity", "viewer psychology", "communication"],
                discussion_style="Direct and practical, focuses on real-world impact and viewer perspective",
                model="vision",
                emoji="👥",
                focus_areas=["audience engagement", "accessibility", "viewer experience", "communication effectiveness"],
                analysis_approach="User-centered and practical, evaluating content from the audience's perspective"
            )
        ]
        return default_agents
    
    def add_agent(self, agent: CustomAgent) -> bool:
        """Add a new agent to the system"""
        try:
            # Check for duplicate names
            if any(existing.name.lower() == agent.name.lower() for existing in self.agents):
                logger.warning(f"Agent with name '{agent.name}' already exists")
                return False
            
            self.agents.append(agent)
            self._save_agent_configurations()
            logger.info(f"✅ Added agent: {agent.name} ({agent.role})")
            return True
        except Exception as e:
            logger.error(f"Failed to add agent: {e}")
            return False
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent by name"""
        try:
            original_count = len(self.agents)
            self.agents = [agent for agent in self.agents if agent.name.lower() != agent_name.lower()]
            
            if len(self.agents) < original_count:
                self._save_agent_configurations()
                logger.info(f"✅ Removed agent: {agent_name}")
                return True
            else:
                logger.warning(f"Agent '{agent_name}' not found")
                return False
        except Exception as e:
            logger.error(f"Failed to remove agent: {e}")
            return False
    
    def update_agent(self, agent_name: str, updated_agent: CustomAgent) -> bool:
        """Update an existing agent's configuration"""
        try:
            for i, agent in enumerate(self.agents):
                if agent.name.lower() == agent_name.lower():
                    self.agents[i] = updated_agent
                    self._save_agent_configurations()
                    logger.info(f"✅ Updated agent: {agent_name}")
                    return True
            
            logger.warning(f"Agent '{agent_name}' not found for update")
            return False
        except Exception as e:
            logger.error(f"Failed to update agent: {e}")
            return False
    
    def get_agent(self, agent_name: str) -> Optional[CustomAgent]:
        """Get an agent by name"""
        for agent in self.agents:
            if agent.name.lower() == agent_name.lower():
                return agent
        return None
    
    def list_agents(self) -> List[CustomAgent]:
        """Get list of all configured agents"""
        return self.agents.copy()
    
    def configure_agents_interactive(self):
        """Interactive agent configuration"""
        print("\n" + "="*60)
        print("🤖 INTERACTIVE AGENT CONFIGURATION")
        print("="*60)
        
        while True:
            print(f"\nCurrently configured agents: {len(self.agents)}")
            for i, agent in enumerate(self.agents, 1):
                print(f"  {i}. {agent.emoji} {agent.name} ({agent.role})")
            
            print("\nOptions:")
            print("  1. Add new agent")
            print("  2. Edit existing agent")
            print("  3. Remove agent")
            print("  4. Reset to defaults")
            print("  5. Save and continue")
            print("  6. Import agent template")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                self._add_agent_interactive()
            elif choice == "2":
                self._edit_agent_interactive()
            elif choice == "3":
                self._remove_agent_interactive()
            elif choice == "4":
                self._reset_to_defaults()
            elif choice == "5":
                break
            elif choice == "6":
                self._import_agent_template()
            else:
                print("Invalid option. Please try again.")
    
    def _add_agent_interactive(self):
        """Interactive agent creation"""
        print("\n📝 CREATE NEW AGENT")
        print("-" * 30)
        
        try:
            name = input("Agent name: ").strip()
            if not name:
                print("❌ Agent name cannot be empty")
                return
            
            # Check for duplicates
            if any(agent.name.lower() == name.lower() for agent in self.agents):
                print(f"❌ Agent '{name}' already exists")
                return
            
            role = input("Agent role/title: ").strip()
            personality = input("Personality description: ").strip()
            
            print("\nExpertise areas (comma-separated):")
            expertise_input = input("Expertise: ").strip()
            expertise = [area.strip() for area in expertise_input.split(",") if area.strip()]
            
            discussion_style = input("Discussion style: ").strip()
            
            print("\nAvailable models:")
            print("  1. gpt_oss (GPT-OSS-120B) - Balanced analysis")
            print("  2. qwen3 (Qwen3-235B) - Creative interpretation")  
            print("  3. vision (Llama4 Maverick) - Visual analysis")
            print("  4. small (Llama-v3.1-8B) - Fast responses")
            
            model_choice = input("Select model (1-4): ").strip()
            model_map = {
                "1": "gpt_oss",
                "2": "qwen3", 
                "3": "vision",
                "4": "small"
            }
            model = model_map.get(model_choice, "gpt_oss")
            
            emoji = input("Agent emoji (default 🤖): ").strip() or "🤖"
            
            print("\nFocus areas (comma-separated):")
            focus_input = input("Focus areas: ").strip()
            focus_areas = [area.strip() for area in focus_input.split(",") if area.strip()]
            
            analysis_approach = input("Analysis approach: ").strip()
            
            # Create agent
            new_agent = CustomAgent(
                name=name,
                role=role,
                personality=personality,
                expertise=expertise,
                discussion_style=discussion_style,
                model=model,
                emoji=emoji,
                focus_areas=focus_areas,
                analysis_approach=analysis_approach
            )
            
            if self.add_agent(new_agent):
                print(f"✅ Successfully created agent: {emoji} {name}")
            else:
                print("❌ Failed to create agent")
                
        except KeyboardInterrupt:
            print("\n❌ Agent creation cancelled")
        except Exception as e:
            print(f"❌ Error creating agent: {e}")
    
    def _edit_agent_interactive(self):
        """Interactive agent editing"""
        if not self.agents:
            print("❌ No agents to edit")
            return
        
        print("\n✏️ EDIT AGENT")
        print("-" * 20)
        
        # List agents
        for i, agent in enumerate(self.agents, 1):
            print(f"  {i}. {agent.emoji} {agent.name} ({agent.role})")
        
        try:
            choice = int(input("Select agent to edit (number): "))
            if 1 <= choice <= len(self.agents):
                agent = self.agents[choice - 1]
                self._edit_single_agent(agent)
            else:
                print("❌ Invalid selection")
        except (ValueError, KeyboardInterrupt):
            print("❌ Edit cancelled")
    
    def _edit_single_agent(self, agent: CustomAgent):
        """Edit a single agent's properties"""
        print(f"\n✏️ Editing: {agent.emoji} {agent.name}")
        print("-" * 40)
        
        properties = [
            ("name", "Name", agent.name),
            ("role", "Role", agent.role),
            ("personality", "Personality", agent.personality),
            ("expertise", "Expertise", ", ".join(agent.expertise)),
            ("discussion_style", "Discussion Style", agent.discussion_style),
            ("model", "Model", agent.model),
            ("emoji", "Emoji", agent.emoji),
            ("focus_areas", "Focus Areas", ", ".join(agent.focus_areas)),
            ("analysis_approach", "Analysis Approach", agent.analysis_approach)
        ]
        
        for prop_name, display_name, current_value in properties:
            print(f"\n{display_name}: {current_value}")
            new_value = input(f"New {display_name.lower()} (press Enter to keep current): ").strip()
            
            if new_value:
                if prop_name in ["expertise", "focus_areas"]:
                    # Handle comma-separated lists
                    new_list = [item.strip() for item in new_value.split(",") if item.strip()]
                    setattr(agent, prop_name, new_list)
                else:
                    setattr(agent, prop_name, new_value)
        
        # Update the agent
        if self.update_agent(agent.name, agent):
            print(f"✅ Successfully updated agent: {agent.emoji} {agent.name}")
        else:
            print("❌ Failed to update agent")
    
    def _remove_agent_interactive(self):
        """Interactive agent removal"""
        if not self.agents:
            print("❌ No agents to remove")
            return
        
        print("\n🗑️ REMOVE AGENT")
        print("-" * 20)
        
        # List agents
        for i, agent in enumerate(self.agents, 1):
            print(f"  {i}. {agent.emoji} {agent.name} ({agent.role})")
        
        try:
            choice = int(input("Select agent to remove (number): "))
            if 1 <= choice <= len(self.agents):
                agent = self.agents[choice - 1]
                confirm = input(f"Remove {agent.emoji} {agent.name}? (y/N): ").strip().lower()
                if confirm == 'y':
                    if self.remove_agent(agent.name):
                        print(f"✅ Removed agent: {agent.name}")
                    else:
                        print("❌ Failed to remove agent")
                else:
                    print("❌ Removal cancelled")
            else:
                print("❌ Invalid selection")
        except (ValueError, KeyboardInterrupt):
            print("❌ Removal cancelled")
    
    def _reset_to_defaults(self):
        """Reset to default agent configuration"""
        confirm = input("Reset to default agents? This will remove all custom agents (y/N): ").strip().lower()
        if confirm == 'y':
            self.agents = self.create_default_agents()
            self._save_agent_configurations()
            print("✅ Reset to default agents")
        else:
            print("❌ Reset cancelled")
    
    def _import_agent_template(self):
        """Import agent from predefined templates"""
        show_available_templates()
        
        template_name = input("\nEnter template name to import: ").strip().lower()
        
        if template_name in AGENT_TEMPLATES:
            template_agents = load_agent_template(template_name)
            if template_agents:
                # Ask if user wants to replace or merge
                choice = input("Replace all current agents (r) or merge (m)? [m]: ").strip().lower()
                
                if choice == 'r':
                    self.agents = template_agents
                    print(f"✅ Replaced agents with {template_name} template")
                else:
                    # Merge, avoiding duplicates
                    existing_names = {agent.name.lower() for agent in self.agents}
                    new_agents = [agent for agent in template_agents if agent.name.lower() not in existing_names]
                    self.agents.extend(new_agents)
                    print(f"✅ Added {len(new_agents)} new agents from {template_name} template")
                
                self._save_agent_configurations()
            else:
                print("❌ Failed to load template")
        else:
            print("❌ Invalid template name")
    
    def _load_agent_configurations(self):
        """Load agent configurations from file"""
        try:
            if os.path.exists(self.agent_configs_file):
                with open(self.agent_configs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.agents = [CustomAgent.from_dict(agent_data) for agent_data in data.get('agents', [])]
                    logger.info(f"✅ Loaded {len(self.agents)} agent configurations")
            else:
                # Create default agents if no config file exists
                self.agents = self.create_default_agents()
                self._save_agent_configurations()
                logger.info("✅ Created default agent configurations")
        except Exception as e:
            logger.error(f"Failed to load agent configurations: {e}")
            # Fallback to defaults
            self.agents = self.create_default_agents()
    
    def _save_agent_configurations(self):
        """Save agent configurations to file"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "agents": [agent.to_dict() for agent in self.agents]
            }
            with open(self.agent_configs_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved {len(self.agents)} agent configurations")
        except Exception as e:
            logger.error(f"Failed to save agent configurations: {e}")
    
    async def conduct_discussion(self, 
                                video_analysis: Dict[str, Any],
                                num_rounds: int = 3,
                                selected_agents: Optional[List[str]] = None) -> List[AgentDiscussionTurn]:
        """Conduct multi-agent discussion with configured agents"""
        
        if not self.agents:
            logger.error("No agents configured for discussion")
            return []
        
        # Select agents for discussion
        discussion_agents = self.agents
        if selected_agents:
            discussion_agents = [agent for agent in self.agents if agent.name in selected_agents]
        
        if not discussion_agents:
            logger.error("No valid agents selected for discussion")
            return []
        
        logger.info(f"Starting {num_rounds}-round discussion with {len(discussion_agents)} agents")
        
        # Prepare video summary
        video_summary = self._prepare_video_summary(video_analysis)
        
        # Discussion topics based on agent expertise
        topics = self._generate_discussion_topics(discussion_agents, num_rounds)
        
        discussion_turns = []
        
        # Run discussion rounds
        for round_num in range(1, num_rounds + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {round_num}/{num_rounds}")
            logger.info(f"{'='*50}")
            
            # Select topic for this round
            topic = topics[min(round_num - 1, len(topics) - 1)]
            logger.info(f"Topic: {topic}")
            
            # Randomize agent order for natural discussion
            round_agents = discussion_agents.copy()
            if round_num > 0:
                random.shuffle(round_agents)
            
            for agent in round_agents:
                # Generate agent response
                response = await self._generate_agent_response(
                    agent=agent,
                    video_summary=video_summary,
                    current_topic=topic,
                    discussion_history=discussion_turns[-3:],  # Last 3 turns for context
                    round_number=round_num
                )
                
                # Add to discussion history
                turn = AgentDiscussionTurn(
                    agent_name=agent.name,
                    agent_role=agent.role,
                    content=response,
                    timestamp=datetime.now().isoformat(),
                    round_number=round_num,
                    responding_to=discussion_turns[-1].agent_name if discussion_turns else None
                )
                
                discussion_turns.append(turn)
                
                # Log the turn
                logger.info(f"\n{agent.emoji} {agent.name} ({agent.role}):")
                logger.info(f"{response[:200]}...")
                
                # Small delay for API rate limiting
                await self.api_manager.acquire() 
        
        self.discussion_history = discussion_turns
        return discussion_turns
    
    def _generate_discussion_topics(self, agents: List[CustomAgent], num_rounds: int) -> List[str]:
        """Generate discussion topics based on agent expertise"""
        base_topics = [
            "Share your initial analysis and key observations about this video content.",
            "Discuss the most significant aspects from your area of expertise.",
            "Evaluate the effectiveness and quality of the content from your perspective.",
            "What improvements or notable strengths would you highlight?",
            "Provide your final assessment and any recommendations."
        ]
        
        # Customize topics based on agent expertise
        if len(agents) > 0:
            # Incorporate agent focus areas into topics
            agent_focuses = []
            for agent in agents:
                agent_focuses.extend(agent.focus_areas[:2])  # Take top 2 focus areas
            
            if agent_focuses:
                specialized_topics = [
                    f"Analyze the video considering {', '.join(agent_focuses[:3])} and other relevant aspects.",
                    f"Discuss how the content addresses {', '.join(agent_focuses[3:6])} and similar considerations.",
                    f"Evaluate the overall effectiveness in terms of {', '.join(agent_focuses[:2])} and quality."
                ]
                
                # Merge with base topics
                combined_topics = base_topics[:2] + specialized_topics + base_topics[3:]
                return combined_topics[:num_rounds]
        
        return base_topics[:num_rounds]
    
    async def _generate_agent_response(self,
                                      agent: CustomAgent,
                                      video_summary: str,
                                      current_topic: str,
                                      discussion_history: List[AgentDiscussionTurn],
                                      round_number: int) -> str:
        """Generate a response from a specific configured agent"""
        
        # Build context from recent discussion
        recent_discussion = "\n".join([
            f"{turn.agent_name}: {turn.content[:150]}..."
            for turn in discussion_history
        ]) if discussion_history else "This is the beginning of our discussion."
        
        # Create agent-specific prompt incorporating their configuration
        prompt = f"""You are {agent.name}, a {agent.role}.

AGENT PROFILE:
Personality: {agent.personality}
Expertise: {', '.join(agent.expertise)}
Focus Areas: {', '.join(agent.focus_areas)}
Discussion Style: {agent.discussion_style}
Analysis Approach: {agent.analysis_approach}

VIDEO CONTENT TO ANALYZE:
{video_summary}

RECENT DISCUSSION:
{recent_discussion}

CURRENT DISCUSSION TOPIC: {current_topic}

As {agent.name}, provide your perspective on this topic. Stay true to your personality, expertise, and analysis approach. 
Reference specific video content when possible. Keep your response focused and insightful (2-3 paragraphs).
If responding to others, acknowledge their points while adding your unique perspective from your area of expertise."""
        
        try:
            # Use the agent's designated model
            result = await self.client.analyze_text(
                text=prompt,
                model_type=agent.model,
                max_tokens=400
            )
            
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating response for {agent.name}: {e}")
            # Fallback response that maintains agent personality
            return f"As a {agent.role}, I find this content noteworthy from the perspective of {', '.join(agent.focus_areas[:2])}. However, I'm experiencing some difficulty articulating my detailed analysis at the moment. I'd appreciate hearing other perspectives to continue our discussion."
    
    def _prepare_video_summary(self, video_analysis: Dict[str, Any]) -> str:
        """Prepare video summary for agent discussion"""
        summary = f"""Video Analysis Summary:
- Duration: Approximately {video_analysis.get('frame_count', 0) * 5} seconds
- Frames analyzed: {video_analysis.get('frame_count', 0)}
- Subtitles: {'Yes' if video_analysis.get('subtitle_count', 0) > 0 else 'No'}

Key Visual Elements:
"""
        
        # Add frame analysis highlights
        if 'frame_analyses' in video_analysis:
            for i, frame in enumerate(video_analysis['frame_analyses'][:4]):  # Top 4 frames
                summary += f"- Frame at {frame.get('timestamp', i*5):.1f}s: {frame.get('analysis', 'N/A')[:120]}...\n"
        
        # Add subtitle highlights
        if video_analysis.get('subtitle_count', 0) > 0 and 'subtitle_analyses' in video_analysis:
            summary += "\nKey Dialogue/Audio:\n"
            for sub in video_analysis['subtitle_analyses'][:2]:  # Top 2 subtitle segments
                summary += f"- {sub.get('subtitle_range', 'N/A')}: {sub.get('text_analyzed', 'N/A')[:100]}...\n"
        
        # Add overall analysis
        if 'overall_analysis' in video_analysis:
            summary += f"\nOverall Assessment:\n{video_analysis['overall_analysis'][:400]}..."
        
        return summary
    
    def export_agent_configuration(self, filename: Optional[str] = None) -> str:
        """Export current agent configuration to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_config_export_{timestamp}.json"
        
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "export_version": "1.0",
                "total_agents": len(self.agents),
                "agents": [agent.to_dict() for agent in self.agents]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported {len(self.agents)} agents to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to export agent configuration: {e}")
            raise
    
    def import_agent_configuration(self, filename: str, merge: bool = True) -> bool:
        """Import agent configuration from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_agents = [CustomAgent.from_dict(agent_data) for agent_data in data.get('agents', [])]
            
            if not merge:
                # Replace all agents
                self.agents = imported_agents
            else:
                # Merge agents (avoid duplicates)
                existing_names = {agent.name.lower() for agent in self.agents}
                new_agents = [agent for agent in imported_agents if agent.name.lower() not in existing_names]
                self.agents.extend(new_agents)
            
            self._save_agent_configurations()
            logger.info(f"✅ Imported {len(imported_agents)} agents from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import agent configuration: {e}")
            return False
    
    def get_discussion_summary(self) -> Dict[str, Any]:
        """Get summary of the last discussion"""
        if not self.discussion_history:
            return {"error": "No discussion history available"}
        
        # Count contributions by agent
        agent_contributions = {}
        for turn in self.discussion_history:
            agent_contributions[turn.agent_name] = agent_contributions.get(turn.agent_name, 0) + 1
        
        # Get unique rounds
        rounds = list(set(turn.round_number for turn in self.discussion_history))
        
        return {
            "total_turns": len(self.discussion_history),
            "participating_agents": len(agent_contributions),
            "rounds_completed": len(rounds),
            "agent_contributions": agent_contributions,
            "discussion_duration": "N/A",  # Could calculate from timestamps
            "agents_configured": len(self.agents)
        }

# CLI interface for agent configuration
def configure_agents_cli():
    """Command-line interface for agent configuration"""
    print("🤖 Multi-Agent Discussion System Configuration")
    print("=" * 60)
    
    system = ConfigurableMultiAgentDiscussion()
    
    while True:
        print(f"\nCurrent agents: {len(system.agents)}")
        for i, agent in enumerate(system.agents, 1):
            print(f"  {i}. {agent.emoji} {agent.name} ({agent.role})")
        
        print("\nConfiguration Options:")
        print("  1. Interactive agent configuration")
        print("  2. Export current configuration") 
        print("  3. Import configuration from file")
        print("  4. View agent details")
        print("  5. Show available templates")
        print("  6. Load agent template")
        print("  7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            system.configure_agents_interactive()
        elif choice == "2":
            try:
                filename = system.export_agent_configuration()
                print(f"✅ Configuration exported to: {filename}")
            except Exception as e:
                print(f"❌ Export failed: {e}")
        elif choice == "3":
            filename = input("Enter filename to import: ").strip()
            if filename and os.path.exists(filename):
                merge = input("Merge with existing agents? (y/N): ").strip().lower() == 'y'
                if system.import_agent_configuration(filename, merge):
                    print("✅ Configuration imported successfully")
                else:
                    print("❌ Import failed")
            else:
                print("❌ File not found")
        elif choice == "4":
            if system.agents:
                for agent in system.agents:
                    print(f"\n{agent.emoji} {agent.name} ({agent.role})")
                    print(f"  Model: {agent.model}")
                    print(f"  Expertise: {', '.join(agent.expertise[:4])}")
                    print(f"  Focus: {', '.join(agent.focus_areas[:3])}")
                    print(f"  Personality: {agent.personality[:100]}...")
            else:
                print("❌ No agents configured")
        elif choice == "5":
            show_available_templates()
        elif choice == "6":
            show_available_templates()
            template_name = input("\nEnter template name to load: ").strip()
            template_agents = load_agent_template(template_name)
            if template_agents:
                replace = input("Replace all current agents? (y/N): ").strip().lower() == 'y'
                if replace:
                    system.agents = template_agents
                else:
                    existing_names = {agent.name.lower() for agent in system.agents}
                    new_agents = [agent for agent in template_agents if agent.name.lower() not in existing_names]
                    system.agents.extend(new_agents)
                system._save_agent_configurations()
                print(f"✅ Loaded template: {template_name}")
            else:
                print("❌ Failed to load template")
        elif choice == "7":
            break
        else:
            print("Invalid option. Please try again.")

# Test the configurable system
async def test_configurable_agents():
    """Test the configurable agent system"""
    print("="*60)
    print("CONFIGURABLE AGENT SYSTEM TEST")
    print("="*60)
    
    # Initialize system
    system = ConfigurableMultiAgentDiscussion()
    
    # Show current agents
    print(f"\nConfigured agents: {len(system.agents)}")
    for agent in system.agents:
        print(f"  {agent.emoji} {agent.name} ({agent.role}) - {agent.model}")
        print(f"    Expertise: {', '.join(agent.expertise[:3])}")
        print(f"    Focus: {', '.join(agent.focus_areas[:2])}")
    
    # Test template loading
    print(f"\n📋 Testing template loading...")
    film_agents = load_agent_template("film_analysis")
    if film_agents:
        print(f"✅ Loaded film_analysis template with {len(film_agents)} agents:")
        for agent in film_agents:
            print(f"  {agent.emoji} {agent.name} ({agent.role})")
    
    # Create mock video analysis for testing
    mock_analysis = {
        "video_path": "test_video.mp4",
        "frame_count": 3,
        "subtitle_count": 2,
        "frame_analyses": [
            {
                "frame_number": 1,
                "timestamp": 5.0,
                "analysis": "A cinematic establishing shot showing urban landscape with dramatic lighting."
            },
            {
                "frame_number": 2,
                "timestamp": 10.0,
                "analysis": "Close-up character portrait with emotional depth and professional cinematography."
            }
        ],
        "subtitle_analyses": [
            {
                "subtitle_range": "0.0s - 5.0s",
                "text_analyzed": "Character dialogue revealing emotional conflict and narrative tension.",
                "analysis": "Dialogue demonstrates strong character development and thematic resonance."
            }
        ],
        "overall_analysis": "Professional film content with strong cinematographic elements and compelling narrative structure."
    }
    
    # Test discussion with different agent configurations
    print(f"\n💬 Testing agent discussion...")
    
    try:
        discussion = await system.conduct_discussion(
            video_analysis=mock_analysis,
            num_rounds=2
        )
        
        print(f"\n✅ Discussion complete with {len(discussion)} turns")
        for turn in discussion:
            agent = system.get_agent(turn.agent_name)
            emoji = agent.emoji if agent else "🤖"
            print(f"\n{emoji} {turn.agent_name}: {turn.content[:150]}...")
        
        # Test summary
        summary = system.get_discussion_summary()
        print(f"\n📊 Discussion Summary:")
        print(f"  Total turns: {summary['total_turns']}")
        print(f"  Participating agents: {summary['participating_agents']}")
        print(f"  Agent contributions: {summary['agent_contributions']}")
        
    except Exception as e:
        print(f"❌ Discussion test failed: {e}")
    
    print(f"\n✅ Configurable agent system test complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "configure":
            configure_agents_cli()
        elif sys.argv[1] == "test":
            asyncio.run(test_configurable_agents())
        elif sys.argv[1] == "templates":
            show_available_templates()
        else:
            print("Usage: python configurable_agent_system.py [configure|test|templates]")
    else:
        print("Configurable Multi-Agent Discussion System")
        print("Usage:")
        print("  python configurable_agent_system.py configure  # Interactive configuration")
        print("  python configurable_agent_system.py test       # Test the system")
        print("  python configurable_agent_system.py templates  # Show available templates")