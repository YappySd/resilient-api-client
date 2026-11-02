"""Fetch star counts for several public GitHub repositories concurrently.

Run: python examples/github_repos.py
The unauthenticated GitHub API allows only a few requests per hour per IP,
so for real use pass a token in the Authorization header.
"""
import asyncio
import logging

from resilient_client import ApiError, ResilientClient

REPOS = ["python/cpython", "encode/httpx", "aiogram/aiogram", "psf/requests"]


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "resilient-client-demo",
        # "Authorization": "Bearer <your-token>",   # keep tokens in environment variables
    }
    async with ResilientClient(
        "https://api.github.com", rate_per_sec=2, burst=2, headers=headers
    ) as api:

        async def stars(repo: str) -> tuple[str, str]:
            try:
                data = await api.get_json(f"/repos/{repo}")
                return repo, str(data["stargazers_count"])
            except ApiError as exc:
                return repo, f"error: {exc}"

        for repo, result in await asyncio.gather(*(stars(r) for r in REPOS)):
            print(f"{repo:<20} {result}")


if __name__ == "__main__":
    asyncio.run(main())
