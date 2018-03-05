package sneklms

object Main {

  import Base._
  import Lisp._
  import Matches._

  def main(args: Array[String]) = {
    val code = gen(args(0), "gen", "snek")
    println(code)
  }

  def gen(arg: String, dir: String, moduleName: String) = {
    val prog_val = parseExp(arg)
    println(prog_val)

    val driver = new DslDriverC[Int,Int](dir, moduleName) with Compiler {
      def snippet(n: Rep[Int]): Rep[Int] = {
        compile(prog_val)(Map("arg" -> Literal(n))) match {
          case Literal(n: Rep[Int]) => n
        }
      }
    }
    if (driver.gen)
      driver.code
    else
      "Error"
  }
}
