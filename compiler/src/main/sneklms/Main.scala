package sneklms

object TestMain {
  import Base._
  import Lisp._
  import Matches._

  def main(args: Array[String]) {
    val prog_val = parseExp(args(0))
    println(trans(prog_val, Nil))
  }
}
