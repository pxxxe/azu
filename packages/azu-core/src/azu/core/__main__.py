"""
Entry point: launch all three core services as separate processes.
Usage: python -m azu.core   or   azu-core
"""
import multiprocessing
import uvicorn


def _run_api():
    uvicorn.run("azu.core.api.main:app", host="0.0.0.0", port=8000)


def _run_scheduler():
    uvicorn.run("azu.core.scheduler.main:app", host="0.0.0.0", port=8001)


def _run_registry():
    uvicorn.run("azu.core.registry.main:app", host="0.0.0.0", port=8002)


def main():
    processes = [
        multiprocessing.Process(target=_run_registry, name="registry"),
        multiprocessing.Process(target=_run_scheduler, name="scheduler"),
        multiprocessing.Process(target=_run_api, name="api"),
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
