#!/usr/bin/env python3
"""Drive an interactive RunPod SSH session over a PTY.

RunPod's proxy ignores the ssh command argument and always starts an
interactive shell, so we send the command on stdin. To avoid matching the
terminal's echo of our own input, the whole remote script is base64-wrapped:
the echoed line is opaque base64, and the sentinels appear only in the real
output.

Usage: python ssh_pty.py "<remote shell body>" [timeout]
"""
import base64
import os
import pty
import select
import subprocess
import sys
import time

HOST = "8nqw27iwtafktn-64412365@ssh.runpod.io"
KEY = os.path.expanduser("~/.ssh/runpod")
BEG = "BEGq7x3a"
END = "ENDq7x3a"


def run(body, timeout=300):
    master, slave = pty.openpty()
    cmd = [
        "ssh", "-tt",
        "-i", KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        "-o", "LogLevel=ERROR",
        HOST,
    ]
    p = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave,
                         close_fds=True)
    os.close(slave)

    def drain(t):
        end = time.time() + t
        buf = b""
        while time.time() < end:
            r, _, _ = select.select([master], [], [], 0.2)
            if r:
                try:
                    d = os.read(master, 65536)
                except OSError:
                    break
                if not d:
                    break
                buf += d
        return buf

    drain(2.5)
    os.write(master, b"stty -echo; export PS1=''; export PROMPT_COMMAND=''\n")
    drain(0.8)

    script = f"echo {BEG}\n{body}\necho {END}\n"
    b64 = base64.b64encode(script.encode()).decode()
    os.write(master, f"echo {b64} | base64 -d | bash\n".encode())

    out = b""
    start = time.time()
    while END.encode() not in out:
        if time.time() - start > timeout:
            break
        r, _, _ = select.select([master], [], [], 0.3)
        if r:
            try:
                d = os.read(master, 65536)
            except OSError:
                break
            if not d:
                break
            out += d

    try:
        os.write(master, b"exit\n")
    except OSError:
        pass
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
    try:
        os.close(master)
    except OSError:
        pass

    text = out.decode("utf-8", "replace")
    if os.environ.get("RAW"):
        return "[RAW]\n" + text
    text = text.replace("\r", "")
    i = text.find(BEG + "\n")
    j = text.find(END)
    if i == -1 or j == -1 or j < i:
        return text
    seg = text[i + len(BEG) + 1:j]
    return seg.strip("\n")


if __name__ == "__main__":
    remote = sys.argv[1]
    tmo = float(sys.argv[2]) if len(sys.argv) > 2 else 300
    sys.stdout.write(run(remote, tmo))
