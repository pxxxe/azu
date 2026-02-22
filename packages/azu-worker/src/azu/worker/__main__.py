"""
Entry point: start the worker.
Usage: python -m azu.worker   or   azu-worker
"""
import asyncio
from azu.worker.main import MoEWorker


def main():
    worker = MoEWorker()
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
