from trame.app import get_server
from .core import XaitkSaliency
from .ml.models import update_ml_device


def main(server=None, **kwargs):
    # Get or create server
    if server is None:
        server = get_server()

    if isinstance(server, str):
        server = get_server(server)

    # CLI
    server.cli.add_argument(
        "--cpu",
        help="Force usage of CPU even if a GPU is available",
        dest="cpu",
        action="store_true",
    )
    update_ml_device(server.cli.parse_known_args()[0].cpu)

    # Init application
    XaitkSaliency(server)

    # Start server
    server.start(**kwargs)


if __name__ == "__main__":
    main()
