init:
	pip3 install -r requirements.txt

build_compiler:
	(cd compiler; sbt assembly) && (mkdir -p jars) && (cp compiler/target/scala-2.11/sneklms.jar jars/)

run_server:
	java -jar jars/sneklms.jar &

test:
	{ java -jar jars/sneklms.jar & }; \
	pid=$$!; \
	sleep 1; \
	py.test; \
	r=$$?; \
	 kill $$pid; \
	exit $$r
