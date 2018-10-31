from .decorators import lms, ast, stage, staged, lmscompile
from subprocess import Popen
import os
import psutil
import time

cmdline = ["java", "-jar", "compiler/target/scala-2.12/sneklms.jar"]
for pid in psutil.pids():
    p = psutil.Process(pid)
    if "java" in p.name() and p.cmdline() == cmdline:
        break
else:
    f = open('.sneklms_server.log', 'w')
    pid = Popen(cmdline, stdin=None, stdout=f).pid
    time.sleep(1)
    print("Server started with pid {}".format(pid))

