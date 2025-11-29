import json
import os
from jinja2 import Template

def generate_report(input_dir, output_path=None):
    """Generate HTML report from pipeline outputs."""
    if output_path is None:
        output_path = os.path.join(input_dir, "report.html")
    
    # Load data
    with open(os.path.join(input_dir, "metrics.json")) as f:
        metrics = json.load(f)
    
    with open(os.path.join(input_dir, "summary.txt")) as f:
        summary = f.read().strip()
    
    # Check for change detection data
    change_metrics = None
    change_polygons = []
    change_brief = None
    
    # Look for change detection files
    polygons_path = os.path.join(input_dir, "polygons.geojson")
    if os.path.exists(polygons_path):
        with open(polygons_path) as f:
            polygons_data = json.load(f)
            change_polygons = polygons_data.get('features', [])
        
        # Change metrics are in the same metrics.json but with additional fields
        if 'confidence' in metrics:
            change_metrics = metrics
            
            # Generate change brief if nl module available
            try:
                from .nl import make_change_brief
                change_brief = make_change_brief(metrics, polygons_data, confidence=metrics.get('confidence'))
            except ImportError:
                change_brief = summary
    
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), "..", "..", "templates", "report.html.j2")
    with open(template_path) as f:
        template = Template(f.read())
    
    # Render
    html = template.render(
        metrics=metrics, 
        summary=summary,
        change_metrics=change_metrics,
        change_polygons=change_polygons,
        change_brief=change_brief
    )
    
    with open(output_path, "w") as f:
        f.write(html)
    
    return output_path