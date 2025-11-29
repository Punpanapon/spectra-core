def make_change_brief(metrics, polygons, aoi_name=None, confidence=None):
    """Generate natural language brief for change detection results."""
    n_polygons = len(polygons['features'])
    total_area = sum(f['properties']['area_ha'] for f in polygons['features'])
    
    # Confidence band
    conf_band = 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
    
    # Area context
    area_desc = "significant" if total_area > 10 else "moderate" if total_area > 1 else "small"
    
    # Location context
    location = f" in {aoi_name}" if aoi_name else ""
    
    # Main finding
    if n_polygons == 0:
        brief = f"No significant vegetation changes detected{location}. "
    else:
        brief = f"Detected {area_desc} vegetation loss{location}: {n_polygons} change areas totaling {total_area:.1f} hectares. "
    
    # Confidence assessment
    brief += f"Analysis confidence: {conf_band} ({confidence:.2f}). "
    
    # Actionable recommendation
    if n_polygons > 0:
        if conf_band == 'High':
            brief += "Recommend field verification and monitoring within 3-5 days."
        elif conf_band == 'Medium':
            brief += "Suggest additional imagery analysis and potential field check within 5-7 days."
        else:
            brief += "Consider acquiring higher resolution imagery or extending analysis period."
    else:
        brief += "Continue routine monitoring schedule."
    
    return brief