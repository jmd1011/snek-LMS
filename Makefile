init:
	pip3 install -r requirements.txt

build_compiler: kill_server
	(cd compiler; sbt assembly)

kill_server:
	pkill -f sneklms.jar

test:
	make -C gen/ clean; \
	py.test -vv
