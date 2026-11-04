"""tools/race_guard.sh offline: a fake tree whose .venv/bin/python stands in for tune.py and spawns a fake 'server' and
'client' (sleeps). HOME points at the tree, so the guard's wineserver -k is a no-op (killing CarlaUE4 through
wineserver is not covered here)."""
import os
import shutil
import signal
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_PY = """#!/usr/bin/env bash
# args: -u tools/tune.py <label> <runs> ...
LABEL=$3; cd "$(dirname "$0")/../../repo"
sleep 300 & echo $! >> pids        # fake server
sleep 300 & echo $! >> pids        # fake client
case "$FAKE_MODE" in
  collide) sleep 1; echo "major collision of tensity 512.0" > "logs/${LABEL}_1.client.log"; sleep 300 ;;
  finish) sleep 1; exit 7 ;;
  *) sleep 300 ;;
esac
"""


def tree(tmp_path):
    os.makedirs(tmp_path / ".venv" / "bin")
    os.makedirs(tmp_path / "repo" / "tools")
    os.makedirs(tmp_path / "repo" / "logs")
    py = tmp_path / ".venv" / "bin" / "python"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    shutil.copy(os.path.join(ROOT, "tools", "race_guard.sh"), tmp_path / "repo" / "tools")
    return tmp_path / "repo"


def start(repo, mode, tmp_path):
    env = dict(os.environ, FAKE_MODE=mode, HOME=str(tmp_path))
    return subprocess.Popen(["bash", str(repo / "tools" / "race_guard.sh"), "lbl", "1"], env=env)


def fake_pids(repo):
    for _ in range(50):
        if (repo / "pids").exists() and len((repo / "pids").read_text().split()) == 2:
            return [int(p) for p in (repo / "pids").read_text().split()]
        time.sleep(0.1)
    raise AssertionError("fake tune.py did not start")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open("/proc/%d/stat" % pid) as f:  # a zombie is dead
        return f.read().split()[2] != "Z"


def test_collision_aborts_and_kills_group(tmp_path):
    repo = tree(tmp_path)
    g = start(repo, "collide", tmp_path)
    pids = fake_pids(repo)
    assert g.wait(timeout=20) == 3
    assert not any(alive(p) for p in pids)
    assert "ABORTED at first collision" in (repo / "logs" / "lbl.tune.log").read_text()


def test_killing_the_guard_kills_the_race(tmp_path):
    repo = tree(tmp_path)
    g = start(repo, "hang", tmp_path)
    pids = fake_pids(repo)
    g.send_signal(signal.SIGTERM)
    assert g.wait(timeout=20) == 143
    assert not any(alive(p) for p in pids)


def test_propagates_tune_status(tmp_path):
    repo = tree(tmp_path)
    g = start(repo, "finish", tmp_path)
    pids = fake_pids(repo)
    assert g.wait(timeout=20) == 7
    assert not any(alive(p) for p in pids)  # leftover children of the group are cleaned up too


def test_eval_client_stays_in_the_guard_group():
    """GNU timeout puts itself (and the client) in a new process group unless --foreground, so race_guard's group
    kill missed demo.py when the guard was stopped (Task 8g). eval.sh must keep the client in tune.py's group."""
    src = open(os.path.join(ROOT, "tools", "eval.sh")).read()
    assert "timeout --foreground 1800 python -u demo.py" in src
    g = subprocess.Popen(["setsid", "bash", "-c", "timeout --foreground 30 sleep 30 & echo $!; wait"],
                         stdout=subprocess.PIPE, text=True)
    pid = int(g.stdout.readline())
    try:
        assert os.getpgid(pid) == os.getpgid(g.pid)
    finally:
        os.killpg(os.getpgid(g.pid), signal.SIGKILL)
