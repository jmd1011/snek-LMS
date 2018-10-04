package sneklms

import Lisp._
import Base._

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import scala.virtualization.lms.common._

import java.io.{PrintWriter, File}

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer

import lantern._

trait CompilerMini extends DiffApi {
  implicit val pos = implicitly[SourceContext]
  val debug =true
  def printDebug(s: String) = if (debug) System.out.println(s)

  abstract class ValueR {
    def get = this
  }
  case class Base(v: NumR) extends ValueR
  case class Func1[A](v: A => NumR @diff) extends ValueR
  case class Func2[A, B](v: (A => B => NumR @diff)) extends ValueR
  case class Func3[A, B, C](v: (A, B, C) => NumR @diff) extends ValueR
  case class Cons[T](v: T) extends ValueR
  case class LitR[T](v: Rep[T]) extends ValueR
  case class MulR[T](v: Var[T]) extends ValueR
  case class WrapR(var v: ValueR) extends ValueR {
    override def get = v.get
  }
  case class ABase(v: ArrayBuffer[NumR]) extends ValueR
  case class AFunc1[A](v: A => ArrayBuffer[NumR] @diff) extends ValueR
  case class AFunc2[A, B](v: (A, B) => ArrayBuffer[NumR] @diff) extends ValueR

  abstract class Model
  case class Bare(f: NumR => NumR @diff) extends Model
  case class F1Array(f: Rep[Array[Double]] => NumR => NumR @diff) extends Model
  case class F1NumR(f: NumR => NumR => NumR @diff) extends Model
  case class F3Array(f: Rep[Array[Double]] => Rep[Array[Int]] => Rep[Array[Int]] => NumR => NumR @diff) extends Model

  def compileModel(exp: Any)(env: Map[String, ValueR]): Model = {

    val ("def":: (f:String) :: (args: List[String]) :: (body: List[List[Any]]) :: Nil) = exp
    printDebug(s"main body >> $body")

    // now the body part should evaluates to NumR @diff
    def com(exp: Any)(implicit envR: Map[String, ValueR] = Map.empty): ValueR @diff = exp match {

      case "def"::(f:String)::(args:List[String])::(body: List[Any])::r =>
        printDebug(s"def >> $f $args $body $r")

        args match {
          case x1::Nil => { // TODO (Fei Wang): we assume x1 is of type NumR
            val F = (x: NumR) => shift { (k: NumR => Unit) =>
              lazy val func: ((NumR => Unit) => NumR => Unit) = FUNL { (k: NumR => Unit) => (x: NumR) =>
                def sh_func: (NumR => NumR @diff) = (x: NumR) => shift{(k: NumR => Unit) => func(k)(x)}
                RST(k( com(body)(envR + (x1 -> Base(x), f -> Func1(sh_func))) match {case Base(v) => v} ))
              }
              func(k)(x)
            }
            printDebug(s"next >>> $r")
            com(r)(envR + (f -> Func1(F)))
          }

          case x1::x2::Nil => { // TODO (Fei Wang): we assume x1 is type Rep[Int], x2 is type NumR
            val F = (i: Rep[Int]) => (x: NumR) => shift { (k: NumR => Unit) =>
              lazy val func: (Rep[Int] => (NumR => Unit) => NumR => Unit) = FUNL1 { (i: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
                def sh_func: (Rep[Int] => NumR => NumR @diff) = (i: Rep[Int]) => (x: NumR) => shift{(k: NumR => Unit) => func(i)(k)(x)}
                RST(k( com(body)(envR + (x1 -> LitR(i), x2 -> Base(x), f -> Func2(sh_func))) match {case Base(v) => v} ))
              }
              func(i)(k)(x)
            }
            com(r)(envR + (f -> Func2(F)))
          }

          /*case "i"::(x2:String)::Nil => { // TODO: (Fei Wang) We assume that "i" means type Rep[Int]
            val F = { (i: Rep[Int], bb: ArrayBuffer[NumR]) => shift { (k: ArrayBuffer[NumR] => Unit) =>

              lazy val func: Rep[Int] => (ArrayBuffer[NumR] => Unit) => ArrayBuffer[NumR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[NumR] => Unit) => (x: ArrayBuffer[NumR]) =>

                def sh_func: ((Rep[Int], ArrayBuffer[NumR]) => ArrayBuffer[NumR] @diff) = (i: Rep[Int], x: ArrayBuffer[NumR]) => shift {k: (ArrayBuffer[NumR] => Unit) => func(i)(k)(x)}

                val betterName = ABase(bb)

                RST(k{ com(body)(envR + ("i" -> LitR(i), x2 -> betterName, f -> AFunc2(sh_func))) match {case ABase(a) => a} })
              }
              func(i)(k)(bb)
            }}
            printDebug(s"next >> $r")
            com(r)(envR + (f -> AFunc2(F)))
          }

          case x1::x2::x3::Nil => { // TODO: (Fei Wang) this function is wrong, because the F and sh_func should have the same type
            // now we need to stage this function (maybe recursive)
            // TODO: (Fei Wang) Problem! type of F is determined by types of args!!
            val F = { (init: NumR, lch: Rep[Array[Int]], rch: Rep[Array[Int]]) => shift { (k: NumR => Unit) =>

              // stuff in here should return type Unit
              lazy val func: Rep[Int] => (NumR => Unit) => NumR => Unit = FUNl { (i: Rep[Int]) => (k: NumR => Unit) => (x: NumR) =>
                def sh_func = (i: Rep[Int]) => shift {k: (NumR => Unit) => func(i)(k)(x)}
                // TODO: this could very much be wrong (Fei Wang)
                RST{k( com(body)(envR + (x1 -> Base(init), x2 -> LitR(lch), x3 -> LitR(rch))) match {case Base(v) => v} )}
              }
              func(0)(k)(init)
            }}
            printDebug(s"next >>> $r")
            com(r)(envR + (f -> Func3(F)))
          }*/
        }

      case "begin"::seq =>
        printDebug(s"seq >> $seq")
        seq match {
          case x :: Nil => com(x)
          case x :: y :: Nil => com(x); com(y)
          case x :: y :: z :: Nil => com(x); com(y); com(z)
          case _ => shift{(k: ValueR => Unit) => ???}
        }
        /*
        val res = seq.Cps.foldLeft(None: Option[ValueR]){
          //case (agg, "None") => agg
          case (agg, exp) => Some(com(exp))
        }
        res.get
        */

      case "let"::(x: String)::"new"::b =>
        com(b)(envR + (x -> ABase(ArrayBuffer[NumR]())))
      case "let"::(x: String)::a::b =>
        com(b)(envR + (x -> com(a)))

      /*
      case "call"::t =>
        t match {
          case "tensor_randinit"::(dim0:Int)::(dim1:Int)::(dummy:Int)::(scale:Float)::Nil =>
            Base(NumR(Tensor.randinit(dim0, dim1, scale)))
          case "tensor_zeros"::(dim0:Int)::Nil =>
            Base(NumR(Tensor.zeros(dim0)))
          case "tuple"::(x:String)::(y:String)::(z:String)::Nil =>
            val (Base(xx: NumR), Base(yy: NumR), Base(zz: NumR)) = (com(x), com(y), com(z))
            ABase(ArrayBuffer(xx, yy, zz))
          case "new_tuple"::Nil =>
            ABase(ArrayBuffer[NumR]())
          case "tensor"::(x:String)::(y:Int)::Nil =>
            val LitR(array: Rep[Array[Float]]) = com(x)
            Base(NumR(Tensor(array, y)))
          case "append"::(x:String)::(y:String)::Nil =>
            val ABase(xx: ArrayBuffer[NumR]) = com(x)
            val Base(yy: NumR) = com(y)
            xx.append(yy)
            Cons(())
          case (x:String)::"sigmoid"::Nil =>
            val Base(xx: NumR) = com(x)
            Base(xx.sigmoid())
          case (x:String)::"tanh"::Nil =>
            val Base(xx: NumR) = com(x)
            Base(xx.tanh())
          case (x:String)::"exp"::Nil =>
            val Base(xx: NumR) = com(x)
            Base(xx.exp())
          case (x:String)::"sum"::Nil =>
            val Base(xx: NumR) = com(x)
            Base(xx.sum())
          case (x:String)::"log"::Nil =>
            val Base(xx: NumR) = com(x)
            Base(xx.log())
        }

      case "dot"::n::m::Nil =>
        printDebug(s"dot $n, $m")
        val Base(nn: NumR) = com(n)
        val Base(mm: NumR) = com(m)
        Base(nn dot mm)
      */
      case "*"::n::m::Nil =>
        printDebug(s"* $n, $m")
        com(n) match {
          case Base(nn: NumR) =>
            com(m) match {
              case Base(mm: NumR) => Base(nn * mm)
              case LitR(mm: Rep[Float]) => Base(nn * new NumR(mm, var_new(0.0f)))
            }
          case LitR(nn: Rep[Float]) =>
            com(m) match {
              case Base(mm: NumR) => Base(new NumR(nn, var_new(0.0f)) * mm)
              case LitR(mm: Rep[Float]) => LitR(nn * mm)
            }
        }
      case "+"::n::m::Nil =>
        printDebug(s"+ $n, $m")
        com(n) match {
          case Base(nn: NumR) =>
            val Base(mm: NumR) = com(m)
            Base(nn + mm)
          case LitR(nn: Rep[Int]) =>
            com(m) match {
              case LitR(mm: Rep[Int]) => LitR(nn + mm)
              case Cons(mm: Int) => LitR(nn + mm)
            }
        }
      case "-"::n::m::Nil =>
        printDebug(s"- $n, $m")
        val Base(nn: NumR) = com(n)
        val Base(mm: NumR) = com(m)
        Base(nn - mm)
      case "<"::n::m::Nil =>
        printDebug(s"< $n, $m")
        val vn: Rep[Int] = com(n) match {
          case LitR(nn: Rep[Int]) => nn
          case Cons(nn: Int) => nn
        }
        val vm: Rep[Int] = com(m) match {
          case LitR(mm: Rep[Int]) => mm
          case Cons(mm: Int) => mm
        }
        LitR(vn < vm)
        /*
      case "/"::n::m::Nil =>
        printDebug(s"/ $n, $m")
        val Base(nn: NumR) = com(n)
        val Base(mm: NumR) = com(m)
        Base(nn / mm)
      case ">="::n::m::Nil =>
        printDebug(s">= $n, $m")
        val vn: Rep[Int] = com(n) match {
          case LitR(nn: Rep[Int]) => nn
          case Cons(nn: Int) => nn
        }
        val vm: Rep[Int] = com(m) match {
          case LitR(mm: Rep[Int]) => mm
          case Cons(mm: Int) => mm
        }
        LitR(vn >= vm)

      case "array-set"::(array:String)::"data"::(index:String)::(value:Int)::Nil =>
        val Base(arr: NumR) = com(array)
        val LitR(idx: Rep[Int]) = com(index)
        val Cons(vlu: Int) = com(value)
        arr.x.data(idx) = vlu
        Cons(())
      */
      case "if"::c::t::e::Nil =>
        printDebug(s"if is here >> $c; $t; $e")
        val LitR(rc: Rep[Boolean]) = com(c)
        /* var flag: Boolean = true
        reset{ com(t) match {
          case Base(t) => flag = true
          case _ => flag = false
        } } */
        Base(IF(rc){ com(t) match {case Base(t) => t} }{ com(e) match {case Base(e) => e} })

      case "idx"::arr::idx::Nil =>
        com(arr) match {
          case ABase(array: ArrayBuffer[NumR]) =>
            val Cons(i: Int) = com(idx)
            Base(array(i))
          case LitR(array: Rep[Array[Int]]) =>
            val LitR(i: Rep[Int]) = com(idx)
            LitR(array(i))
        }
      case "len"::arr::Nil =>
        com(arr) match {
          case LitR(array: Rep[Array[Double]]) =>
            LitR(array.length)
        }
      case "array-get"::arr::idx::Nil =>
        com(arr) match {
          case LitR(array: Rep[Array[Double]]) =>
            val LitR(i: Rep[Int]) = com(idx)
            LitR(array(i))
        }

      case x: Int => Cons(x)
      case x: String =>
        printDebug(s"EnvR >> x > $x")
        envR(x)
      case x::Nil =>
        printDebug(s"single >> x > $x")
        com(x)

      case f::(x: List[Any]) =>
        printDebug(s"f >> $f")
        printDebug(s"x >> $x")
        val nf = com(f)
        printDebug(s"nf >> $nf")
        (nf, x) match {
          case (Func1(f: (NumR => NumR @diff)), a::Nil) =>
            com(a) match {
              case Base(aa: NumR) => Base(f(aa))
            }
          // TODO: (Fei Wang) this case is shadowed by the case above !!!! Try other methods??
          case (Func1(f: (Rep[Int] => NumR @diff)), a::Nil) =>
            com(a) match {
              case LitR(aa: Rep[Int]) => Base(f(aa))
              case Cons(aa: Int) => Base(f(aa))
            }
          case (Func2(f: (Rep[Int] => NumR => NumR @diff)), a::b::Nil) =>
            printDebug(s"in function >> nf > $f, x > $x")
            val Base(bb: NumR) = com(b)
            com(a) match {
              case LitR(aa: Rep[Int]) =>
                printDebug(s"before application >> nf > $f, aa > $aa, bb > $bb")
                Base(f(aa)(bb))
              case Cons(aa: Int) =>
                printDebug(s"before application >> nf > $f, aa > $aa, bb > $bb")
                Base(f(aa)(bb))
            }
        }

      case todo => {printDebug(s"todo>>>$todo"); shift{(k: ValueR => Unit) => ???} }
    }

    // TODO: (Fei Wang): this is assuming the knowledge about the types of args
    if (args.size == 1) {
      // we are assuming that the only argument is NumR type
      Bare { x: NumR =>
        val envR = env + (args(0) -> Base(x))
        com(body)(envR) match { case Base(v) => v }
      }
    } else if (args.size == 2) {
      // we are assuming that the first argument is Rep[Array[Double]], and the second is NumR
      // F1Array { (arr: Rep[Array[Double]]) => (x: NumR) =>
      //   com(body)(env + (args(0) -> LitR(arr), args(1) -> Base(x))) match { case Base(v) => v }
      // }
      F1NumR { (base: NumR) => (x: NumR) =>
        com(body)(env + (args(0) -> Base(base), args(1) -> Base(x))) match {case Base(v) => v}
      }
    } else { // assuming that the args size is 4
      // we are assuming that the args are of type Rep[Array[Double]], Rep[Array[Int]], Rep[Array[Int]], and NumR respectively
      F3Array { (values: Rep[Array[Double]]) => (lchs: Rep[Array[Int]]) => (rchs: Rep[Array[Int]]) => (x: NumR) =>
        com(body)(env + (args(0) -> LitR(values), args(1) -> LitR(lchs), args(2) -> LitR(rchs), args(3) -> Base(x))) match {case Base(v) => v}
      }
    }
  }
}

