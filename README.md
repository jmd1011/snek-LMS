# snek-LMS
LMS in Python

[![Build Status](https://travis-ci.org/jmd1011/snek-LMS.svg?branch=master)](https://travis-ci.org/jmd1011/snek-LMS)

# Compile and run the S-expr to C compiler

First you need to have swig install.

    make build_compiler             # compile the compiler
    java -jars compiler/target/scala-2.11/sneklms.jar &    # start the server (maybe better to do it in an other shell rather than in background)
    python3 compiler/pipeline.py    # simple python script calling the compiler
