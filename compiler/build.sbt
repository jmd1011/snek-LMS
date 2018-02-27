name := "snek-lms"

version := "1.0"

organization := "Purdue"



/**
 *  The main sbt configuration for the compiler build.
 */

lazy val defaults = Seq(
  // scala compiler version:
  scalaVersion := "2.11.12",
  scalaBinaryVersion := "2.11",
  scalacOptions ++= Seq("-feature", "-deprecation", "-unchecked"),

  libraryDependencies += "org.scala-lang.modules" %% "scala-parser-combinators" % "1.1.0",

  // source location configuration:
  scalaSource in Compile := baseDirectory.value / "src",
  scalaSource in Test := baseDirectory.value / "src/test",
  resourceDirectory in Compile := baseDirectory.value / "resources",

  // eclipse configuration:
  unmanagedSourceDirectories in Compile := Seq((scalaSource in Compile).value),
  unmanagedSourceDirectories in Test := Seq((scalaSource in Test).value),

  // run configuration
  fork in run := true,
  connectInput in run := false,
  outputStrategy := Some(StdoutOutput),
  javaOptions in run ++= Seq("-Xss32M", "-Xms128M"),

  // test configuration:
  libraryDependencies += "com.novocode" % "junit-interface" % "0.11" % "test",
  testOptions += Tests.Argument(TestFrameworks.JUnit, "-q", "-v"),
  parallelExecution in Test := false,

  // fork in test
  fork in Test := true,
  javaOptions in Test ++= Seq("-Xss32M", "-Xms128M")
)

// The location desired for the source jar that will be submitted
lazy val snek = (project in file(".")).settings(
  defaults,
  name := "snek-lms")
