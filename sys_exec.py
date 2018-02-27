import subprocess

# Executes the given args in a new shell
def exec_command_line(args):
    subprocess.run(args, shell=True)


