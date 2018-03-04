init:
	pip3 install -r requirements.txt

build_compiler:
	(cd compiler; sbt assembly) && (cp compiler/target/scala-2.11/sneklms.jar jars/)

run_server:
	java -jar jars/sneklms.jar &

test:
	py.test
