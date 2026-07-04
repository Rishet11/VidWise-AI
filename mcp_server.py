"""Command-line entry point for the VidWise MCP server."""
from vidwise_mcp.server import mcp


if __name__ == "__main__":
    mcp.run()
