import logging

import click

__all__ = ["edit"]

logger = logging.getLogger("utoolbox.cli.dataset")


@click.group()
@click.pass_context
def edit(ctx):
    """Edit the dataset internals."""


@edit.command()
def crop(ctx):
    pass


@edit.command()
def metadata(ctx):
    pass
