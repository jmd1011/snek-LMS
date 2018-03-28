name := "snek-lms"

version := "1.0"

organization := "Purdue"

scalaVersion := "2.12.4"

// scalaOrganization := "org.scala-lang.virtualized"

// cps

autoCompilerPlugins := true

addCompilerPlugin("org.scala-lang.plugins" % "scala-continuations-plugin_2.12.0" % "1.0.3")

libraryDependencies += "org.scala-lang.plugins" % "scala-continuations-library_2.12" % "1.0.3"

scalacOptions += "-P:continuations:enable"



resolvers += Resolver.sonatypeRepo("snapshots")
resolvers += Resolver.mavenLocal

lazy val lantern = (project in file("Lantern"))
lazy val root = (project in file(".")).dependsOn(lantern % "compile")

libraryDependencies += "org.scala-lang.lms" %% "lms-core-macrovirt" % "0.9.0-SNAPSHOT"

libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang.modules" %% "scala-parser-combinators" % "1.1.0"

libraryDependencies += "org.apache.commons" % "commons-lang3" % "3.4"

libraryDependencies += "net.sf.py4j" % "py4j" % "0.8.1"

val paradiseVersion = "2.1.0"

addCompilerPlugin("org.scalamacros" % "paradise" % paradiseVersion cross CrossVersion.full)

// scalacOptions += "-Yvirtualize"

scalacOptions += "-deprecation"

assemblyJarName in assembly := "sneklms.jar"
mainClass in assembly := Some("server.ServerApplication")
// assemblyMergeStrategy in assembly := {
//   case PathList("javax", "servlet", xs @ _*)         => MergeStrategy.first
//   case PathList(ps @ _*) if ps.last endsWith ".html" => MergeStrategy.first
//   case "application.conf"                            => MergeStrategy.concat
//   case "unwanted.txt"                                => MergeStrategy.discard
//   case x =>
//     val oldStrategy = (assemblyMergeStrategy in assembly).value
//     oldStrategy(x)
// }
