init:
	pip3 install -r requirements.txt

data:
	python3 gen_data.py

build_compiler: kill_server
	(cd compiler; sbt assembly)

process := $(shell pgrep -f sneklms.jar)

kill_server:
ifneq ($(process),)
	pkill -f sneklms.jar
else
	@echo "Server not running"
endif

test:
	make -C gen/ clean; \
	py.test -vv
