package sneklms

object Main {

  import Base._
  import Lisp._
  import Matches._

  def main(args: Array[String]) = {
    val prog_val = parseExp(args(0))

    val code = new DslDriverC[Int,Int] with Compiler {
      def snippet(n: Rep[Int]): Rep[Int] = {
        compile(prog_val).asInstanceOf[Rep[Int]]
      }
    }.code

    println(code)
  }
}
