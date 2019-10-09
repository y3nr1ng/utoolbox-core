import logging
import os, errno

import click
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
from pyftpdlib.authorizers import DummyAuthorizer
import pyotp


@click.command()
@click.argument("root")
def main(root):
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Instantiate a dummy authorizer for managing 'virtual' users
    authorizer = DummyAuthorizer()

    # Define a new user having full r/w permissions and a read-only
    totp = pyotp.TOTP("deadbeef")
    passcode = totp.now()
    print(f"\n Please enter your one-time passcode [ {passcode} ] from client.\n")
    authorizer.add_user("user", passcode, root, perm="elradfmwMT")

    # Instantiate FTP handler class
    handler = FTPHandler
    handler.authorizer = authorizer

    # Define a customized banner (string returned when client connects)
    handler.banner = "pyftpdlib based ftpd ready."

    # Specify a masquerade address and the range of ports to use for
    # passive connections.  Decomment in case you're behind a NAT.
    # handler.masquerade_address = '151.25.42.11'
    # handler.passive_ports = range(60000, 65535)

    # Instantiate FTP server class and listen on 0.0.0.0:2121
    address = ("10.109.20.7", 2121)
    server = FTPServer(address, handler)

    # set a limit for connections
    server.max_cons = 256

    # start ftp server
    server.serve_forever(timeout=30)


if __name__ == "__main__":
    import coloredlogs

    logging.getLogger("pyftpdlib").setLevel(logging.INFO)

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    main()
