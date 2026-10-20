"""One worker process for the Lecture 11 crash run: python worker.py HOST:PORT"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

import repair_wf as wf


async def main(target):
    client = await Client.connect(target)
    worker = Worker(client, task_queue="repair",
                    workflows=[wf.RepairNaive, wf.RepairKeyed],
                    activities=[wf.call_model, wf.run_tests_naive, wf.run_tests_keyed],
                    activity_executor=ThreadPoolExecutor(4))
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
