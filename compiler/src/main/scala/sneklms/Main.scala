package sneklms

object Main {

  import Base._
  import Lisp._
  import Matches._

  def main(args: Array[String]) = {
    val code = compileMain(args(0))
    println(code)
  }

  def compileMain(arg: String) = {
    val prog_val = parseExp(arg)
    println(prog_val)

    val code = new DslDriverC[Int,Int] with Compiler {
      def snippet(n: Rep[Int]): Rep[Int] = {
        compile(prog_val)(Map("arg" -> Literal(n))) match {
          case Literal(n: Rep[Int]) => n
        }
      }
    }.code

    code
  }
}
