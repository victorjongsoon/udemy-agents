from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("date-server")

@mcp.tool()
def get_current_date() -> str:
    """"Return the current date."""
    return datetime.now().strftime("%Y-%m-%d")

if __name__ == "__main__":
    mcp.run()