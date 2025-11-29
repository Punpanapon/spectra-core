import click
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from spectra_core.ai.stac_fetch import main as stac_fetch_main
from spectra_core.ai.chipper import main as chipper_main
from spectra_core.ai.make_dummy_label import main as make_dummy_label_main

@click.group()
def cli():
    """AI training data tools."""
    pass

@cli.command('stac-fetch')
@click.pass_context
def stac_fetch(ctx):
    """Fetch Sentinel data via STAC."""
    ctx.forward(stac_fetch_main)

@cli.command('make-chips')
@click.pass_context
def make_chips(ctx):
    """Generate training chips from raster data."""
    ctx.forward(chipper_main)

@cli.command('make-dummy-label')
@click.pass_context
def make_dummy_label(ctx):
    """Create dummy labels from NDVI thresholding."""
    ctx.forward(make_dummy_label_main)

if __name__ == '__main__':
    cli()