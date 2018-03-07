package sneklms

import Lisp._
import Base._

import scala.lms.common._
import scala.reflect.SourceContext

import java.io.{PrintWriter, File}


trait Compiler extends Dsl {


  abstract class Value {
    def get = this
  }
  case class Literal[T](v: Rep[T]) extends Value
  case class Mut[T](v: Var[T]) extends Value
  case class Wrap(var v: Value) extends Value {
    override def get = v.get
  }
  case object VError extends Value

  type Env = Map[String,Value]

  def compile[T,U](n: Any, m: Any)(op: (Rep[T], Rep[T]) => Rep[U])(implicit env: Env): Value = (compile(n), compile(m)) match {
    case (Literal(n: Rep[T]), Literal(m: Rep[T])) => Literal(op(n, m))
  }

  implicit def repToValue[T](x: Rep[T]) = Literal(x)
  val debug = false
  def printDebug(s: String) = if (debug) System.out.println(s)

  def printEnv(implicit env: Env) = {
    printDebug("====== Env =======")
    env foreach { case (k, v) => printDebug(s"$k -> $v") }
    printDebug("==================")
  }

  def compile(exp: Any)(implicit env: Env = Map.empty): Value = { printDebug(s"exp >> $exp"); exp } match {
    case "None" => unit(-1)
    case "new" => Mut(var_new(0))
    case "set"::(x: String)::a::Nil =>
      val Mut(vx: Var[Int]) = env(x)
      var_assign(vx, compile(a) match { case Literal(a: Rep[Int]) => a })
      unit(())
    case "get"::(x: String)::Nil =>
      val Mut(vx: Var[Int]) = env(x)
      Literal(readVar(vx))
    case "while"::t::body::Nil =>
      while (compile(t) match { case Literal(t: Rep[Boolean]) => t })
        compile(body) match { case Literal(b: Rep[Unit]) => b }
      unit(())
    case x: Int => unit(x)
    case x: String => env(x)
    case Str(x) => Literal(unit(x))
    case "*"::n::m::Nil =>
      compile[Int,Int](n, m)(_ * _)
    case "+"::n::m::Nil =>
      compile[Int,Int](n, m)(_ + _)
    case "-"::n::m::Nil =>
      compile[Int,Int](n, m)(_ - _)
    case "=="::n::m::Nil =>
      compile[Int,Boolean](n, m)(_ == _)
    case "<"::n::m::Nil =>
      compile[Int,Boolean](n, m)(_ < _)
    case "if"::c::t::e::Nil =>
      val Literal(rc: Rep[Boolean]) = compile(c)
      Literal(if (rc) compile(t) match { case Literal(t: Rep[Int]) => t } else compile(e) match { case Literal(e: Rep[Int]) => e })
    case "let"::(x: String)::a::b =>
      compile(b)(env + (x -> compile(a)))
    case "return"::x::Nil =>
      val Literal(rx: Rep[Any]) = compile(x)
      return rx
    case "print"::x::Nil =>
      val arg = compile(x) match { case Literal(x: Rep[String]) => x }
      printf("%s\\n", arg)
      unit(1)
    case "call"::t =>
      t match {
        case "numpy"::"zeros"::x::Nil =>
          compile(x) match {
            case Literal(x: Rep[Int]) => NewArray[Int](x)
          }
      }
    case "def"::(f: String)::(args: List[String])::(body: List[List[Any]])::r =>
      printDebug(s"body >> $body")
      val func = args match {
        case x1::Nil =>
          lazy val fptr: Rep[Int => Int] = uninlinedFunc1 { (x1v: Rep[Int]) =>
            compile(body)(env + (x1 -> Literal(x1v)) + (f -> Literal(fptr))) match {
              case Literal(n: Rep[Int]) => n
            }
          }
          Literal(fptr)
        case x1::x2::Nil =>
          lazy val fptr: Rep[(Int, Int) => Int] = uninlinedFunc2 { (x1v: Rep[Int], x2v: Rep[Int]) =>
            compile(body)(env + (x1 -> Literal(x1v)) + (x2 -> Literal(x2v)) + (f -> Literal(fptr))) match {
              case Literal(n: Rep[Int]) => n
            }
          }
          Literal(fptr)
      }
      compile(r)(env + (f -> func))
    case "lambda"::(f: String)::(x: String)::e::Nil =>
      lazy val fptr: Rep[Int => Int] = fun { (xv: Rep[Int]) =>
        compile(e)(env + (x -> Literal(xv)) + (f -> Literal(fptr))) match {
          case Literal(n: Rep[Int]) => n
        }
      }
      Literal(fptr)
    case "begin"::seq =>
      printDebug(s"seq >> $seq")
      val res = ((None: Option[Value]) /: seq) {
        case (agg, exp) => Some(compile(exp))
      }
      res.get
    case x::Nil =>
      compile(x)
    case f::(x: List[Any]) =>
      printDebug(s"f >> $f")
      printDebug(s"x >> $x")
      val args = x map(compile(_) match { case Literal(x: Rep[Int]) => x })
      printDebug(s"args >> $args")
      val nf = compile(f).get
      printEnv
      printDebug(s"nf >> $nf")
      (nf, args) match {
        case (Literal(f: Rep[Int => Int]), x1::Nil) => f(x1)
        case (Literal(f: Rep[((Int, Int)) => Int]), x1::x2::Nil) => f((x1, x2))
      }
    case Nil => // no main
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

  def uninlinedFunc0[B:Typ](f: Function0[Rep[B]]): Rep[Unit=>B]
  def uninlinedFunc1[A:Typ,B:Typ](f: Rep[A]=>Rep[B])(implicit pos: SourceContext): Rep[A => B]
  def uninlinedFunc2[A1:Typ,A2:Typ,B:Typ](f: Function2[Rep[A1],Rep[A2],Rep[B]]): Rep[(A1,A2)=>B]
  implicit def funTyp2[A1:Typ,A2:Typ,B:Typ]: Typ[(A1,A2) => B]
  def uninlinedFunc3[A1:Typ,A2:Typ,A3:Typ,B:Typ](f: Function3[Rep[A1],Rep[A2],Rep[A3],Rep[B]]): Rep[(A1,A2,A3)=>B]
  implicit def funTyp3[A1:Typ,A2:Typ,A3:Typ,B:Typ]: Typ[(A1,A2,A3) => B]
}

trait DslExp extends Dsl with PrimitiveOpsExpOpt with NumericOpsExpOpt with BooleanOpsExp with IfThenElseExpOpt with EqualExpBridgeOpt with RangeOpsExp with OrderingOpsExp with MiscOpsExp with EffectExp with ArrayOpsExpOpt with StringOpsExp with SeqOpsExp with FunctionsRecursiveExp with WhileExp with StaticDataExp with VariablesExpOpt with ObjectOpsExpOpt with MathOpsExp with UncheckedOpsExp with TupledFunctionsExp {

  case class UninlinedFunc0[B:Typ](b: Block[B]) extends Def[Unit => B] {
    val mB = typ[B]
  }
  case class UninlinedFunc1[A:Typ,B:Typ](s:Sym[A], b: Block[B]) extends Def[A => B] {
    val mA = typ[A]
    val mB = typ[B]
  }
  case class UninlinedFunc2[A1:Typ,A2:Typ,B:Typ](s1:Sym[A1], s2:Sym[A2], b: Block[B]) extends Def[(A1,A2) => B] {
    val mA1 = typ[A1]
    val mA2 = typ[A2]
    val mB = typ[B]
  }
  implicit def funTyp2[A1:Typ,A2:Typ,B:Typ]: Typ[(A1,A2) => B] = {
    implicit val ManifestTyp(mA1) = typ[A1]
    implicit val ManifestTyp(mA2) = typ[A2]
    implicit val ManifestTyp(mB) = typ[B]
    manifestTyp
  }
  case class UninlinedFunc3[A1:Typ,A2:Typ,A3:Typ,B:Typ](s1:Sym[A1], s2:Sym[A2], s3:Sym[A3], b: Block[B]) extends Def[(A1,A2,A3) => B] {
    val mA1 = typ[A1]
    val mA2 = typ[A2]
    val mA3 = typ[A3]
    val mB = typ[B]
  }
  implicit def funTyp3[A1:Typ,A2:Typ,A3:Typ,B:Typ]: Typ[(A1,A2,A3) => B] = {
    implicit val ManifestTyp(mA1) = typ[A1]
    implicit val ManifestTyp(mA2) = typ[A2]
    implicit val ManifestTyp(mA3) = typ[A3]
    implicit val ManifestTyp(mB) = typ[B]
    manifestTyp
  }
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
    override def isPrimitiveType[T](m: Typ[T]) = (m == typ[String]) || super.isPrimitiveType(m)

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

    // BEGINNING UNINLINED FUNCTIONS
    val functionList0 = new scala.collection.mutable.HashMap[Sym[Any],Block[Any]]()
    val functionList1 = new scala.collection.mutable.HashMap[Sym[Any],(Sym[Any],Block[Any])]()
    val functionList2 = new scala.collection.mutable.HashMap[Sym[Any],(Sym[Any],Sym[Any],Block[Any])]()
    val functionList3 = new scala.collection.mutable.HashMap[Sym[Any],(Sym[Any],Sym[Any],Sym[Any],Block[Any])]()
    def uninlinedFunc0[B:Typ](f: Function0[Rep[B]]) = {
      val b = reifyEffects(f())
      uninlinedFunc0(b)
    }
    def uninlinedFunc0[B:Typ](b: Block[B]) = {
      val l = reflectEffect(UninlinedFunc0(b), Pure())
      functionList0 += (l.asInstanceOf[Sym[Any]] -> b)
      l
    }

    val topfunTable = new scala.collection.mutable.HashMap[Any,Sym[Any]]()
    def uninlinedFunc1[A:Typ,B:Typ](f: Exp[A] => Exp[B])(implicit pos: SourceContext): Exp[A => B] = {
      val can = canonicalize(f)
      topfunTable.get(can) match {
        case Some(funSym) =>
          funSym.asInstanceOf[Exp[A=>B]]
        case _ =>
          val funSym = fresh[A=>B]
          topfunTable += can->funSym.asInstanceOf[Sym[Any]]
          val s = fresh[A]
          val b = reifyEffects(f(s))
          functionList1 += (funSym.asInstanceOf[Sym[Any]] -> (s,b))
          funSym
      }
    }
    def canonicalize1(f: Any) = {
      val s = new java.io.ByteArrayOutputStream()
      val o = new java.io.ObjectOutputStream(s)
      o.writeObject(f)
      s.toString("ASCII")
    }

    def uninlinedFunc2[A1:Typ,A2:Typ,B:Typ](f: (Rep[A1],Rep[A2])=>Rep[B]) = {
      val can = canonicalize1(f)
      topfunTable.get(can) match {
        case Some(funSym) =>
          funSym.asInstanceOf[Exp[(A1,A2)=>B]]
        case _ =>
          val funSym = fresh[(A1,A2)=>B]
          topfunTable += can->funSym.asInstanceOf[Sym[Any]]
          val s1 = fresh[A1]
          val s2 = fresh[A2]
          val b = reifyEffects(f(s1,s2))
          functionList2 += (funSym.asInstanceOf[Sym[Any]] -> (s1,s2,b))
          funSym
      }
    }
    def uninlinedFunc2[A1:Typ,A2:Typ,B:Typ](s1: Sym[A1], s2: Sym[A2], b: Block[B]) = {
      // val l = reflectEffect(UninlinedFunc2(s1,s2,b), Pure())
      // functionList2 += (l.asInstanceOf[Sym[Any]] -> (s1,s2,b))
      // l
      ???
    }

    def uninlinedFunc3[A1:Typ,A2:Typ,A3:Typ,B:Typ](f: Function3[Rep[A1],Rep[A2],Rep[A3],Rep[B]]) = {
      // val s1 = fresh[A1]
      // val s2 = fresh[A2]
      // val s3 = fresh[A3]
      // val b = reifyEffects(f(s1,s2,s3))
      // uninlinedFunc3(s1,s2,s3,b)
      ???
    }
    def uninlinedFunc3[A1:Typ,A2:Typ,A3:Typ,B:Typ](s1: Sym[A1], s2: Sym[A2], s3: Sym[A3], b: Block[B]) = {
      // val l = reflectEffect(UninlinedFunc3(s1,s2,s3,b), Pure())
      // functionList3 += (l.asInstanceOf[Sym[Any]] -> (s1,s2,s3,b))
      // l
      ???
    }
    override def doLambdaDef[A:Typ,B:Typ](f: Exp[A] => Exp[B]) : Def[A => B] = {
      val x = unboxedFresh[A]
      val y = reifyEffects(f(x)) // unfold completely at the definition site.
      ???

      Lambda(f, x, y)
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
  def tmpremap[A](m: Typ[A]): String = m.toString match {
    case "Int" => "int"
    case _ => remap(m)
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
    case UninlinedFunc2(_, _, _) => {}
    case _ => super.emitNode(sym,rhs)
  }

  var dir: String = _
  var moduleName: String = _

  override def emitSource[A:Typ](args: List[Sym[_]], body: Block[A], functionName: String, out: PrintWriter) = {

    val sA = remap(typ[A])

    withStream(out) {
      stream.println(s"""/*****************************************/
       |#include <stdio.h>
       |#include <stdlib.h>
       |#include <stdint.h>
       |#include "$moduleName.h"
       |using namespace std;""".stripMargin)

      emitFunctions(dir, moduleName)

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
  def emitFunctions(dir: String, moduleName: String) = {
    // Output prototypes to resolve dependencies
    withStream(new PrintWriter(s"$dir/$moduleName.h")) {
      stream.println("#ifndef _code")
      stream.println("#define _code")
      functionList0.foreach(f=>stream.println(tmpremap(getBlockResult(f._2).tp) + " " + quote(f._1) + "();"))
      functionList1.foreach(f=>stream.println(tmpremap(getBlockResult(f._2._2).tp) + " " + quote(f._1) + "(" + tmpremap(f._2._1.tp) + " " + quote(f._2._1) + ");"))
      functionList2.foreach(f=>stream.println(tmpremap(getBlockResult(f._2._3).tp) + " " + quote(f._1) + "(" + tmpremap(f._2._1.tp) + " " + quote(f._2._1) + ", " + tmpremap(f._2._2.tp) + " " + quote(f._2._2) +");\n"))
      functionList3.foreach(f=>stream.println(remap(getBlockResult(f._2._4).tp) + " " + quote(f._1) + "(" + remap(f._2._1.tp) + " " + quote(f._2._1) + ", " + remap(f._2._2.tp) + " " + quote(f._2._2) + ", " + remap(f._2._3.tp) + " " + quote(f._2._3) + ");\n"))
      stream.println("#endif")
    }
    withStream(new PrintWriter(s"$dir/$moduleName.i")) {
      stream.println(s"%module $moduleName")
      stream.println("%{")
      stream.println(s"""#include "$moduleName.h"""")
      stream.println("  #include <stdlib.h>")
      stream.println("  #include <stdio.h>")
      stream.println("  #include <stdint.h>")
      stream.println("%}")
      stream.println(s"""%include "$moduleName.h"""")
    }

    // Output actual functions
    functionList0.foreach(func => {
      stream.println(tmpremap(getBlockResult(func._2).tp) + " " + quote(func._1) + "() {")
      emitBlock(func._2)
      stream.println("return " + quote(getBlockResult(func._2)) + ";")
      stream.println("}\n")
    })
    functionList1.foreach(func => {
      stream.print(tmpremap(getBlockResult(func._2._2).tp) + " " + quote(func._1) + "(")
      stream.print(tmpremap(func._2._1.tp) + " " + quote(func._2._1))
      stream.println(") {")
      emitBlock(func._2._2)
      stream.println("return " + quote(getBlockResult(func._2._2)) + ";")
      stream.println("}\n")
    })
    functionList2.foreach(func => {
      stream.print(tmpremap(getBlockResult(func._2._3).tp) + " " + quote(func._1) + "(")
      stream.print(tmpremap(func._2._1.tp) + " " + quote(func._2._1) + ", ")
      stream.print(tmpremap(func._2._2.tp) + " " + quote(func._2._2))
      stream.println(") {")
      emitBlock(func._2._3)
      stream.println("return " + quote(getBlockResult(func._2._3)) + ";")
      stream.println("}\n")
    })
    functionList3.foreach(func => {
      stream.print(tmpremap(getBlockResult(func._2._4).tp) + " " + quote(func._1) + "(")
      stream.print(tmpremap(func._2._1.tp) + " " + quote(func._2._1) + ", ")
      stream.print(tmpremap(func._2._2.tp) + " " + quote(func._2._2) + ", ")
      stream.print(tmpremap(func._2._3.tp) + " " + quote(func._2._3))
      stream.println(") {")
      emitBlock(func._2._4)
      stream.println("return " + quote(getBlockResult(func._2._4)) + ";")
      stream.println("}\n")
    })
    functionList0.clear
    functionList1.clear
    functionList2.clear
    functionList3.clear
  }
}


abstract class DslSnippet[A:Manifest,B:Manifest] extends Dsl {
  def snippet(x: Rep[A]): Rep[B]
}

abstract class DslDriverC[A:Manifest,B:Manifest](ddir: String, mmoduleName: String) extends DslSnippet[A,B] with DslExp { q =>
  val codegen = new DslGenC {
    val IR: q.type = q
  }

  codegen.dir = ddir
  codegen.moduleName = mmoduleName

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
    codegen.emitSource(snippet, "entrypoint", new PrintWriter(source))

    indent(source.toString)
  }

  def gen = {
    val file = new PrintWriter(s"$ddir/$mmoduleName.cpp")
    file.println(code)
    file.flush
    file.close
    import scala.sys.process._
    System.out.println(s"======= Compile module $mmoduleName ============")
    try {
      (s"make MODULE_NAME=$mmoduleName -C $ddir":ProcessBuilder).lines.foreach(Console.println _)
      System.out.println(s"========================================")
      true
    } catch {
      case _ => false
    }
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

