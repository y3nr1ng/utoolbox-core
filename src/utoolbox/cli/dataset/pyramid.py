import logging

import click

__all__ = ["pyramid"]

logger = logging.getLogger("utoolbox.cli.dataset")


@click.group()
@click.pass_context
def pyramid(ctx):
    """Modify pyramid info stored in the dataset."""


@pyramid.command()
@click.pass_context
def create(ctx):
    pass


@pyramid.command()
@click.pass_context
def delete(ctx):
    pass
