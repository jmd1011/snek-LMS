name := "snek-lms"

organization := "Purdue"

scalaVersion := "2.11.2"

scalaOrganization := "org.scala-lang.virtualized"


resolvers += Resolver.sonatypeRepo("snapshots")

libraryDependencies += "org.scala-lang.lms" %% "lms-core" % "1.0.1-SNAPSHOT"

libraryDependencies += "org.scala-lang.virtualized" % "scala-compiler" % "2.11.2"

libraryDependencies += "org.scala-lang.virtualized" % "scala-library" % "2.11.2"

libraryDependencies += "org.scala-lang.virtualized" % "scala-reflect" % "2.11.2"

libraryDependencies += "org.scala-lang.modules" %% "scala-parser-combinators" % "1.1.0"

scalacOptions += "-Yvirtualize"

scalacOptions += "-deprecation"

