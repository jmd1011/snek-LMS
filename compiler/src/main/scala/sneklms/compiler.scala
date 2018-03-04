package sneklms

import Lisp._
import Base._

import scala.lms.common._
import scala.reflect.SourceContext


trait Compiler extends Dsl {


  abstract class Value {
    def get = this
  }
  case class Literal[T](v: Rep[T]) extends Value
  case class Wrap(var v: Value) extends Value {
    override def get = v.get
  }
  case object VError extends Value

  type Env = Map[String,Value]

  def compile[T,U](n: Any, m: Any)(op: (Rep[T], Rep[T]) => Rep[U])(implicit env: Env): Value = (compile(n), compile(m)) match {
    case (Literal(n: Rep[T]), Literal(m: Rep[T])) => Literal(op(n, m))
  }

  implicit def repToValue[T](x: Rep[T]) = Literal(x)

  def printEnv(implicit env: Env) = {
    System.out.println("====== Env =======")
    env foreach { case (k, v) => System.out.println(s"$k -> $v") }
    System.out.println("==================")
  }

  def compile(exp: Any)(implicit env: Env = Map.empty): Value = exp match {
    case x: Int => unit(x)
    case x: String => env getOrElse (x, Literal(unit(-42)))
    case x::Nil => compile(x)
    case "*"::n::m =>
      compile[Int,Int](n, m)(_ * _)
    case "+"::n::m =>
      compile[Int,Int](n, m)(_ + _)
    case "-"::n::m =>
      compile[Int,Int](n, m)(_ - _)
    case "=="::n::m =>
      compile[Int,Boolean](n, m)(_ == _)
    case "if"::c::t::e =>
      val Literal(rc: Rep[Boolean]) = compile(c)
      compile[Int,Int](t, e) { (t: Rep[Int], e: Rep[Int]) =>
        if (rc) t else e
      }
    case "let"::(x: String)::a::b =>
      compile(b)(env + (x -> compile(a)))
    case "return"::x =>
      val Literal(rx: Rep[Int]) = compile(x)
      return rx
    case "def"::(f: String)::(args: List[String])::body::r =>
      val func = args match {
        case x1::Nil =>
          lazy val fptr: Rep[Int => Int] = fun { (x1v: Rep[Int]) =>
            compile(body)(env + (x1 -> Literal(x1v)) + (f -> Literal(fptr))) match {
              case Literal(n: Rep[Int]) => n
            }
          }
          Literal(fptr)
        case x1::x2::Nil =>
          lazy val fptr: Rep[((Int, Int)) => Int] = fun { (x1v: Rep[Int], x2v: Rep[Int]) =>
            compile(body)(env + (x1 -> Literal(x1v)) + (x2 -> Literal(x2v)) + (f -> Literal(fptr))) match {
              case Literal(n: Rep[Int]) => n
            }
          }
          Literal(fptr)
      }
      compile(r)(env + (f -> func))
    case "lambda"::(f: String)::(x: String)::e =>
      lazy val fptr: Rep[Int => Int] = fun { (xv: Rep[Int]) =>
        compile(e)(env + (x -> Literal(xv)) + (f -> Literal(fptr))) match {
          case Literal(n: Rep[Int]) => n
        }
      }
      Literal(fptr)
    case f::(x: List[Any]) =>
      val args = x map(compile(_) match { case Literal(x: Rep[Int]) => x })
      (compile(f).get, args) match {
        case (Literal(f: Rep[Int => Int]), x1::Nil) => f(x1)
        case (Literal(f: Rep[((Int, Int)) => Int]), x1::x2::Nil) => f((x1, x2))
      }
    case Nil =>
      val x = unit(0)
      return x
  }
}

trait Dsl extends PrimitiveOps with NumericOps with BooleanOps with LiftString with LiftPrimitives with LiftNumeric with LiftBoolean with IfThenElse with Equal with RangeOps with OrderingOps with MiscOps with ArrayOps with StringOps with SeqOps with Functions with While with StaticData with Variables with LiftVariables with ObjectOps with TupledFunctions {
  implicit def repStrToSeqOps(a: Rep[String]) = new SeqOpsCls(a.asInstanceOf[Rep[Seq[Char]]])
  override def infix_&&(lhs: Rep[Boolean], rhs: => Rep[Boolean])(implicit pos: scala.reflect.SourceContext): Rep[Boolean] =
    __ifThenElse(lhs, rhs, unit(false))
  def generate_comment(l: String): Rep[Unit]
  def comment[A:Typ](l: String, verbose: Boolean = true)(b: => Rep[A]): Rep[A]
}

trait DslExp extends Dsl with PrimitiveOpsExpOpt with NumericOpsExpOpt with BooleanOpsExp with IfThenElseExpOpt with EqualExpBridgeOpt with RangeOpsExp with OrderingOpsExp with MiscOpsExp with EffectExp with ArrayOpsExpOpt with StringOpsExp with SeqOpsExp with FunctionsRecursiveExp with WhileExp with StaticDataExp with VariablesExpOpt with ObjectOpsExpOpt with MathOpsExp with UncheckedOpsExp with TupledFunctionsExp {
  override def boolean_or(lhs: Exp[Boolean], rhs: Exp[Boolean])(implicit pos: SourceContext) : Exp[Boolean] = lhs match {
    case Const(false) => rhs
    case _ => super.boolean_or(lhs, rhs)
  }
  override def boolean_and(lhs: Exp[Boolean], rhs: Exp[Boolean])(implicit pos: SourceContext) : Exp[Boolean] = lhs match {
    case Const(true) => rhs
    case _ => super.boolean_and(lhs, rhs)
  }

  case class GenerateComment(l: String) extends Def[Unit]
  def generate_comment(l: String) = reflectEffect(GenerateComment(l))
  case class Comment[A:Typ](l: String, verbose: Boolean, b: Block[A]) extends Def[A]
  def comment[A:Typ](l: String, verbose: Boolean)(b: => Rep[A]): Rep[A] = {
    val br = reifyEffects(b)
    val be = summarizeEffects(br)
    reflectEffect[A](Comment(l, verbose, br), be)
  }

  override def boundSyms(e: Any): List[Sym[Any]] = e match {
    case Comment(_, _, b) => effectSyms(b)
    case _ => super.boundSyms(e)
  }

  override def array_apply[T:Typ](x: Exp[Array[T]], n: Exp[Int])(implicit pos: SourceContext): Exp[T] = (x,n) match {
    case (Def(StaticData(x:Array[T])), Const(n)) =>
      val y = x(n)
      if (y.isInstanceOf[Int]) unit(y) else staticData(y)
    case _ => super.array_apply(x,n)
  }

  // TODO: should this be in LMS?
  override def isPrimitiveType[T](m: Typ[T]) = (m == manifest[String]) || super.isPrimitiveType(m)

  override def doApply[A:Typ,B:Typ](f: Exp[A => B], x: Exp[A])(implicit pos: SourceContext): Exp[B] = {
    val x1 = unbox(x)
    val x1_effects = x1 match {
      case UnboxedTuple(l) => l.foldLeft(Pure())((b,a)=>a match {
        case Def(Lambda(_, _, yy)) => b orElse summarizeEffects(yy)
        case _ => b
      })
        case _ => Pure()
    }
    f match {
      case Def(Lambda(_, _, y)) => reflectEffect(Apply(f, x1), summarizeEffects(y) andAlso x1_effects)
      case _ => reflectEffect(Apply(f, x1), Simple() andAlso x1_effects)
    }
  }

}

// TODO: currently part of this is specific to the query tests. generalize? move?
trait DslGenC extends CGenNumericOps
    with CGenPrimitiveOps with CGenBooleanOps with CGenIfThenElse
    with CGenEqual with CGenRangeOps with CGenOrderingOps
    with CGenMiscOps with CGenArrayOps with CGenStringOps
    with CGenSeqOps with CGenFunctions with CGenWhile
    with CGenStaticData with CGenVariables
    with CGenObjectOps with CGenUncheckedOps with CLikeGenMathOps
    with CGenTupledFunctions {
  val IR: DslExp
  import IR._

  def getMemoryAllocString(count: String, memType: String): String = {
      "(" + memType + "*)malloc(" + count + " * sizeof(" + memType + "));"
  }
  override def remap[A](m: Typ[A]): String = m.toString match {
    case "java.lang.String" => "char*"
    case "Array[Char]" => "char*"
    case "Char" => "char"
    case _ => super.remap(m)
  }
  override def format(s: Exp[Any]): String = {
    remap(s.tp) match {
      case "uint16_t" => "%c"
      case "bool" | "int8_t" | "int16_t" | "int32_t" => "%d"
      case "int64_t" => "%ld"
      case "float" | "double" => "%f"
      case "string" => "%s"
      case "char*" => "%s"
      case "char" => "%c"
      case "void" => "%c"
      case _ =>
        import scala.lms.internal.GenerationFailedException
        throw new GenerationFailedException("CGenMiscOps: cannot print type " + remap(s.tp))
    }
  }

  // we treat string as a primitive type to prevent memory management on strings
  // strings are always stack allocated and freed automatically at the scope exit
  override def isPrimitiveType(tpe: String) : Boolean = {
    tpe match {
      case "char*" => true
      case "char" => true
      case _ => super.isPrimitiveType(tpe)
    }
  }

  override def quote(x: Exp[Any]) = x match {
    case Const(s: String) => "\""+s.replace("\"", "\\\"")+"\"" // TODO: more escapes?
    case Const('\n') if x.tp == typ[Char] => "'\\n'"
    case Const('\t') if x.tp == typ[Char] => "'\\t'"
    case Const(0)    if x.tp == typ[Char] => "'\\0'"
    case _ => super.quote(x)
  }
  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case a@ArrayNew(n) =>
      val arrType = remap(a.m)
      stream.println(arrType + "* " + quote(sym) + " = " + getMemoryAllocString(quote(n), arrType))
    case ArrayApply(x,n) => emitValDef(sym, quote(x) + "[" + quote(n) + "]")
    case ArrayUpdate(x,n,y) => stream.println(quote(x) + "[" + quote(n) + "] = " + quote(y) + ";")
    case PrintLn(s) => stream.println("printf(\"" + format(s) + "\\n\"," + quoteRawString(s) + ");")
    case StringCharAt(s,i) => emitValDef(sym, "%s[%s]".format(quote(s), quote(i)))
    case Comment(s, verbose, b) =>
      stream.println("//#" + s)
      if (verbose) {
        stream.println("// generated code for " + s.replace('_', ' '))
      } else {
        stream.println("// generated code")
      }
      emitBlock(b)
      emitValDef(sym, quote(getBlockResult(b)))
      stream.println("//#" + s)
    case _ => super.emitNode(sym,rhs)
  }
  override def emitSource[A:Typ](args: List[Sym[_]], body: Block[A], functionName: String, out: java.io.PrintWriter) = {

    val sA = remap(typ[A])

    withStream(out) {
      stream.println("""/*****************************************/
       |#include <stdio.h>
       |#include <stdlib.h>
       |#include <stdint.h>
       |using namespace std;""".stripMargin)

      stream.println(sA+" "+functionName+"("+args.map(a => remapWithRef(a.tp)+" "+quote(a)).mkString(", ")+") {")
      emitBlock(body)

      val y = getBlockResult(body)
      if (remap(y.tp) != "void")
        stream.println("return " + quote(y) + ";")

      stream.println("}")
      stream.println("/*******************************************/")
      }
      Nil
    }
  }


  abstract class DslSnippet[A:Manifest,B:Manifest] extends Dsl {
    def snippet(x: Rep[A]): Rep[B]
  }

  abstract class DslDriverC[A:Manifest,B:Manifest] extends DslSnippet[A,B] with DslExp { q =>
    val codegen = new DslGenC {
      val IR: q.type = q
    }

    def indent(str: String) = {
      val strLines = str.split("\n")
      val res = new StringBuilder
      var level: Int = 0
      for (line <- strLines) {
        if(line.contains("}")) level -= 1
        res ++= (("  " * level) + line + "\n")
      if(line.contains("{")) level += 1
      }
      res.toString
    }

    lazy val code: String = {
      implicit val mA = manifestTyp[A]
      implicit val mB = manifestTyp[B]
      val source = new java.io.StringWriter()
      codegen.emitSource(snippet, "entrypoint", new java.io.PrintWriter(source))

      indent(source.toString)
    }
    def eval(a:A): Unit = { // TBD: should read result of type B?
      val out = new java.io.PrintWriter("/tmp/snippet.c")
      out.println(code)
      out.close
      //TODO: use precompile
      (new java.io.File("/tmp/snippet")).delete
      import scala.sys.process._
      (s"gcc -std=c99 -O3 /tmp/snippet.c -o /tmp/snippet":ProcessBuilder).lines.foreach(Console.println _)
      (s"/tmp/snippet $a":ProcessBuilder).lines.foreach(Console.println _)
    }
  }

