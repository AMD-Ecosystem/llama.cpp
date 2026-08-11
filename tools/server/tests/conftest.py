import os
import pytest
from utils import *


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    # the gpu-rocm MTP check runs a single self-contained test on an SSL-less
    # build, so skip preset pre-caching (it fetches over the binary's HTTPS).
    if os.environ.get("GG_MTP_GREEDY") == "1":
        return
    # this will be run once per test session, before any tests
    ServerPreset.load_all()
