import os
import json
import click
from ..ai.paths import open_da
from .insights import summarize_ndvi, insight_bullets, narrative

@click.command()
@click.option('--red', required=True, help='RED band path')
@click.option('--nir', required=True, help='NIR band path')
@click.option('--out', required=True, help='Output directory')
def main(red, nir, out):
    """Generate insights from RED/NIR bands."""
    os.makedirs(out, exist_ok=True)
    
    print(f"Loading {red} and {nir}...")
    red_da = open_da(red)
    nir_da = open_da(nir)
    
    # Align grids if needed
    if red_da.rio.crs != nir_da.rio.crs or red_da.shape != nir_da.shape:
        nir_da = nir_da.rio.reproject_match(red_da)
    
    print("Computing NDVI summary...")
    summary = summarize_ndvi(red_da, nir_da)
    
    # Save outputs
    with open(f"{out}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    bullets = insight_bullets(summary)
    with open(f"{out}/bullets.txt", 'w') as f:
        f.write('\n'.join(f"• {bullet}" for bullet in bullets))
    
    story = narrative(summary)
    with open(f"{out}/narrative.txt", 'w') as f:
        f.write(story)
    
    print(f"Insights saved to {out}/")
    print(f"Summary: {len(bullets)} bullets, {len(story.split())} word narrative")

if __name__ == '__main__':
    main()