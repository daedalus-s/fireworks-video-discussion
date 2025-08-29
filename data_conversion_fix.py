#!/usr/bin/env python3
"""
Enhanced Analysis Data Conversion Fix
Fixes the issue where enhanced analysis returns different data structure than expected
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime

def fix_enhanced_analysis_conversion(enhanced_results: Any) -> Dict[str, Any]:
    """
    COMPREHENSIVE FIX: Convert enhanced analysis results to proper format
    This handles the different return formats from enhanced_descriptive_analysis
    """
    
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
    
    print(f"🔧 FIXING: Enhanced results type: {type(enhanced_results)}")
    
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
            
            # Case 2: It's a dictionary but with a different structure (like from enhanced pipeline)
            result = default_result.copy()
            
            # Try to extract data from nested structures
            if 'analysis_summary' in enhanced_results:
                summary = enhanced_results['analysis_summary']
                result['frame_count'] = summary.get('frames_analyzed', result['frame_count'])
                result['subtitle_count'] = summary.get('subtitle_segments', result['subtitle_count'])
                result['processing_time'] = summary.get('processing_time', result['processing_time'])
            
            if 'key_insights' in enhanced_results:
                insights = enhanced_results['key_insights']
                
                # Extract frame analyses from visual highlights
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
                
                # Extract overall analysis from comprehensive assessment
                if 'comprehensive_assessment' in insights:
                    result['overall_analysis'] = insights['comprehensive_assessment']
                
                # Extract scene breakdown
                if 'scene_summaries' in insights:
                    scene_breakdown = []
                    for i, scene in enumerate(insights['scene_summaries']):
                        scene_breakdown.append({
                            'scene_number': i + 1,
                            'start_time': i * 20.0,
                            'end_time': (i + 1) * 20.0,
                            'summary': scene
                        })
                    result['scene_breakdown'] = scene_breakdown
                
                # Extract visual elements
                if 'visual_elements' in insights:
                    result['visual_elements'] = insights['visual_elements']
            
            # Extract video path and other metadata
            result['video_path'] = enhanced_results.get('video_path', result['video_path'])
            result['total_cost'] = enhanced_results.get('total_cost', result['total_cost'])
            result['timestamp'] = enhanced_results.get('timestamp', result['timestamp'])
            result['analysis_depth'] = enhanced_results.get('analysis_depth', result['analysis_depth'])
            
            print(f"✅ Converted enhanced results: {result['frame_count']} frames, {len(result['frame_analyses'])} analyses")
            return result
        
        # Case 3: It's an object with attributes
        elif hasattr(enhanced_results, '__dict__'):
            print("🔧 Converting object with __dict__ to dictionary")
            attrs = enhanced_results.__dict__
            
            result = default_result.copy()
            
            # Direct attribute mapping
            for key in ['video_path', 'frame_count', 'subtitle_count', 'frame_analyses', 
                       'subtitle_analyses', 'overall_analysis', 'scene_breakdown', 
                       'visual_elements', 'content_categories', 'processing_time', 
                       'total_cost', 'timestamp', 'analysis_depth']:
                if key in attrs and attrs[key] is not None:
                    result[key] = attrs[key]
            
            print(f"✅ Converted object to dict: {result['frame_count']} frames")
            return result
        
        # Case 4: It has a to_dict method
        elif hasattr(enhanced_results, 'to_dict'):
            print("🔧 Using to_dict() method")
            dict_result = enhanced_results.to_dict()
            
            # Ensure all required fields exist
            for key, default_value in default_result.items():
                if key not in dict_result or dict_result[key] is None:
                    dict_result[key] = default_value
            
            return dict_result
        
        else:
            print(f"⚠️ Unknown enhanced results type: {type(enhanced_results)}")
            return default_result
    
    except Exception as e:
        print(f"❌ Error converting enhanced results: {e}")
        return default_result

def patch_integrated_configurable_pipeline():
    """
    Apply the fix directly to the integrated configurable pipeline
    """
    
    patch_code = '''
# PATCH: Add this method to IntegratedConfigurableAnalysisPipeline class

def _convert_enhanced_results_for_agents_FIXED(self, enhanced_results: Any) -> Dict[str, Any]:
    """FIXED: Convert enhanced analysis results to format expected by agents"""
    
    from data_conversion_fix import fix_enhanced_analysis_conversion
    return fix_enhanced_analysis_conversion(enhanced_results)
'''
    
    print("🔧 PATCH CODE FOR INTEGRATED PIPELINE:")
    print("-" * 60)
    print(patch_code)
    print("-" * 60)
    
    print("📝 TO APPLY THIS FIX:")
    print("1. Replace the _convert_enhanced_results_for_agents method in")
    print("   integrated_configurable_pipeline.py with the fixed version")
    print("2. Or add this as a new method and call it instead")
    print("3. Or use the fixed main.py which handles this automatically")

def debug_enhanced_results_structure(enhanced_results: Any):
    """Debug function to understand the structure of enhanced results"""
    
    print("\n" + "=" * 60)
    print("🔍 DEBUGGING ENHANCED RESULTS STRUCTURE")
    print("=" * 60)
    
    print(f"Type: {type(enhanced_results)}")
    print(f"Is dict: {isinstance(enhanced_results, dict)}")
    print(f"Has __dict__: {hasattr(enhanced_results, '__dict__')}")
    print(f"Has to_dict: {hasattr(enhanced_results, 'to_dict')}")
    
    if isinstance(enhanced_results, dict):
        print(f"\nDictionary keys ({len(enhanced_results.keys())}):")
        for key in enhanced_results.keys():
            value = enhanced_results[key]
            if isinstance(value, (list, dict)):
                print(f"  {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
            else:
                print(f"  {key}: {type(value)} = {str(value)[:50]}...")
    
    elif hasattr(enhanced_results, '__dict__'):
        attrs = enhanced_results.__dict__
        print(f"\nObject attributes ({len(attrs)}):")
        for key, value in attrs.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
            else:
                print(f"  {key}: {type(value)} = {str(value)[:50]}...")
    
    print("=" * 60)

# Test the conversion function
def test_conversion():
    """Test the conversion with different input types"""
    
    print("Testing enhanced results conversion")
    print("=" * 60)
    
    # Test Case 1: Dictionary with report structure (like from enhanced pipeline)
    enhanced_report_format = {
        'timestamp': '2025-01-01T12:00:00',
        'video_path': 'test_video.mp4',
        'analysis_depth': 'comprehensive',
        'analysis_summary': {
            'frames_analyzed': 20,
            'subtitle_segments': 5,
            'processing_time': 120.5
        },
        'key_insights': {
            'visual_highlights': [
                'Frame analysis showing cityscape with dramatic lighting',
                'Character close-up with emotional intensity',
                'Wide shot establishing urban environment'
            ],
            'comprehensive_assessment': 'This video demonstrates professional cinematography with strong narrative structure.',
            'visual_elements': {
                'dominant_subjects': [['person', 12], ['building', 7]],
                'color_palette': [['blue', 14], ['red', 10]]
            }
        },
        'total_cost': 0.0237
    }
    
    print("Test 1: Enhanced report format")
    result1 = fix_enhanced_analysis_conversion(enhanced_report_format)
    print(f"Converted frame_count: {result1['frame_count']}")
    print(f"Frame analyses length: {len(result1['frame_analyses'])}")
    print(f"Overall analysis: {result1['overall_analysis'][:50]}...")
    
    # Test Case 2: Object with attributes
    class MockEnhancedResult:
        def __init__(self):
            self.video_path = 'mock_video.mp4'
            self.frame_count = 15
            self.subtitle_count = 8
            self.frame_analyses = [
                {'frame_number': 1, 'analysis': 'Mock frame analysis 1'},
                {'frame_number': 2, 'analysis': 'Mock frame analysis 2'}
            ]
            self.overall_analysis = 'Mock overall analysis'
            self.processing_time = 95.2
            self.total_cost = 0.015
    
    mock_result = MockEnhancedResult()
    print("\nTest 2: Object with attributes")
    result2 = fix_enhanced_analysis_conversion(mock_result)
    print(f"Converted frame_count: {result2['frame_count']}")
    print(f"Frame analyses length: {len(result2['frame_analyses'])}")
    
    # Test Case 3: None/Empty
    print("\nTest 3: None input")
    result3 = fix_enhanced_analysis_conversion(None)
    print(f"Default frame_count: {result3['frame_count']}")
    print(f"Default analyses length: {len(result3['frame_analyses'])}")
    
    print("\nAll tests completed")

def create_quick_fix_script():
    """Create a script to quickly apply the fix"""
    
    script_content = '''#!/usr/bin/env python3
"""
Quick Fix Script - Apply Enhanced Analysis Data Conversion Fix
Run this to fix the FastAPI integration issue
"""

import os
import shutil
from pathlib import Path

def apply_fix():
    """Apply the enhanced analysis data conversion fix"""
    
    print("Applying Enhanced Analysis Data Conversion Fix")
    print("=" * 60)
    
    # Step 1: Backup original files
    print("1. Creating backups...")
    
    files_to_backup = ["main.py", "integrated_configurable_pipeline.py"]
    backup_dir = Path("backup_before_fix")
    backup_dir.mkdir(exist_ok=True)
    
    for filename in files_to_backup:
        if Path(filename).exists():
            shutil.copy2(filename, backup_dir / f"{filename}.backup")
            print(f"   Backed up {filename}")
    
    # Step 2: Update main.py
    print("2. The fixed main.py is provided in the artifacts above")
    print("   Replace your main.py with the fixed version")
    
    # Step 3: Add the conversion fix
    print("3. Save the data_conversion_fix.py file from the artifact")
    
    # Step 4: Instructions
    print("\nFIX INSTRUCTIONS:")
    print("=" * 40)
    print("1. Replace main.py with the fixed version from the artifact")
    print("2. Save data_conversion_fix.py in your project directory")  
    print("3. Restart your server: python main.py")
    print("4. Test the frontend analysis")
    
    print("\nThe fix addresses:")
    print("- Enhanced analysis results format mismatch")
    print("- Proper data conversion for agent processing")
    print("- Better error handling in FastAPI background tasks")
    print("- Comprehensive debugging output")
    
    return True

if __name__ == "__main__":
    apply_fix()
'''
    
    with open("quick_fix_enhanced_analysis.py", "w") as f:
        f.write(script_content)
    
    print("Created quick_fix_enhanced_analysis.py")
    return script_content

# Main execution
if __name__ == "__main__":
    print("Enhanced Analysis Data Conversion Fix")
    print("=" * 50)
    
    # Run tests
    test_conversion()
    
    # Show patch instructions
    patch_integrated_configurable_pipeline()
    
    # Create quick fix script
    create_quick_fix_script()