from trame import get_cli_parser
from pathlib import Path

# Initialize
parser = get_cli_parser()
parser.add_argument(
    "--cpu",
    help="Force usage of CPU even if a GPU is available",
    dest="cpu",
    action="store_true",
)


def get_args():
    parser = get_cli_parser()
    args, _ = parser.parse_known_args()
    return args
