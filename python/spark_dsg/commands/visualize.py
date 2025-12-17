"""Entry point for visualizing scene graph."""

import time

import click
import spark_dsg as dsg
from spark_dsg.viser import ViserRenderer


@click.command("visualize")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--ip", default="localhost")
@click.option("--port", default="8080")
@click.option("--image-prefix", default=None, help="Root search directory for images using O_<id> convention.")
def cli(filepath, ip, port, image_prefix):
    """Visualize a scene graph from FILEPATH using Open3D."""
    G = dsg.DynamicSceneGraph.load(filepath)

    with ViserRenderer(ip, port=port, image_folder_prefix=image_prefix) as renderer:
        renderer.draw(G)
        while True:
            time.sleep(10.0)
