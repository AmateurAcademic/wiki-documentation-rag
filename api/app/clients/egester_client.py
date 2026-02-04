import httpx

class EgesterClient:
    """Client for interacting with the Egester Note Writer API."""
    def __init__(self, base_url: str, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds)

    async def write_note(self, file_name: str, content: str, append: bool) -> None:
        """Send a request to the Note Writer API to write or append a note."""
        url = f"{self.base_url}/write_note"
        payload = {
            "file_name": file_name,
            "content": content,
            "append": append
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()