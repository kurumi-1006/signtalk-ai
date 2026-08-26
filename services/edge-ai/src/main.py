import argparse
import logging

from .config.settings import Settings
from .serve import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, help='Override the configured HTTP port when serving.')
    args = parser.parse_args()
    settings = Settings.model_validate({})
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    if args.serve:
        if args.port:
            settings = settings.model_copy(update={'edge_port': args.port})
        import uvicorn
        uvicorn.run(create_app(settings), host=settings.edge_host, port=settings.edge_port)
        return
    parser.error('Use --serve to start the recognition service.')
if __name__ == '__main__':
    main()
