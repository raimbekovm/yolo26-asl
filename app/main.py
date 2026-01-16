"""Entry point for Gradio application."""

import argparse


def launch(share: bool = False, server_port: int = 7860) -> None:
    """Launch the Gradio application."""
    from app.gradio_app import create_app

    app = create_app()
    app.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0",
    )


def main():
    parser = argparse.ArgumentParser(description="Launch ASL Recognition App")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")

    args = parser.parse_args()
    launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
