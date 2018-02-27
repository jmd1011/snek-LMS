import subprocess
from functools import reduce

COMPILER = "gcc"
OUTPUT_FLAG = "-o"

# Executes the given args in a new shell
def exec_command_line(command):
    subprocess.run(command, shell=True)

# takes an array and creates string of elements separated with spaces
def gen_command(arr):
    if(arr == []): return ""
    return reduce((lambda acc, el: acc + " " + el), arr)

# Takes a c file name, an array of options for GCC, and an output file name
# Runs the GCC compiler on the file
def compile_c_file(file_name, options, output_file):
    string_command = gen_command([   
            COMPILER,                   # gcc
            file_name,                  # file.c
            gen_command(options),       # -g or so
            OUTPUT_FLAG, output_file    # -o output.o
        ])
    exec_command_line(string_command)

def run_output_file(output_file):
    command = "./" + output_file
    exec_command_line(command)


# EXAMPLE: 
compile_c_file("./test.c", [], "output")
run_output_file("output")