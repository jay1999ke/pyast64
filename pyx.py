import argparse
import ast
from enum import Enum
import sys
from typing import Callable, List, TextIO
from astpretty import pprint as pp


class Emitter(object):
    """Base class for anything that can emit assembly code."""

    def Emit(self) -> str:
        """
        Return the assembly representation of the object.
        """
        raise NotImplementedError("Emit not implemented")


class Register(Emitter, Enum):
    """
    Enum for x86-64 general-purpose registers with assembly emit support.
    """

    RAX = "rax"  # Accumulator register
    RBX = "rbx"  # Base register
    RCX = "rcx"  # Counter register
    RDX = "rdx"  # Data register
    RSI = "rsi"  # Source index
    RDI = "rdi"  # Destination index
    R8 = "r8"    # General purpose register 8
    R9 = "r9"    # General purpose register 9
    RSP = "rsp"  # Stack pointer
    RBP = "rbp"  # Base pointer (frame pointer)

    def Emit(self):
        return "%{}".format(self.value)


class Literal(Emitter):

    def __init__(self, value: int) -> None:
        self.value: int = value

    def Emit(self) -> str:
        return "${}".format(self.value)


class Dereference(Emitter):
    """
    Represents a memory dereference operation on a register,
    optionally with an integer offset.

    This models assembly addressing modes like:
    - (register)           : dereference at the address in the register
    - offset(register)     : dereference at address (register + offset)
    """

    def __init__(self, register: Register) -> None:
        self.register: Register = register
        self.offset: int = None

    def WithOffset(self, offset: int) -> Emitter:
        """
        Set an offset for the dereference, returning the updated instance.

        Args:
            offset (int): The offset to add to the base register address.
        """
        self.offset = offset
        return self

    def Emit(self) -> str:
        offset: str = "{}".format(self.offset) if self.offset else ""
        return "{offset}({register})".format(offset=offset, register=self.register.Emit())


class OpCode(Emitter, Enum):
    PUSHQ = "pushq"  # Push quadword onto stack
    POPQ = "popq"    # Pop quadword from stack
    MOVQ = "movq"    # Move quadword
    LEAQ = "leaq"    # Load effective address quadword
    ADDQ = "addq"    # Add quadword
    SUBQ = "subq"    # Subtract quadword
    IMULQ = "imulq"  # Integer multiply quadword
    CQO = "cqo"      # Convert quadword to octword (sign extend)
    IDIV = "idiv"    # Integer divide
    AND = "and"      # Bitwise AND
    OR = "or"        # Bitwise OR
    XOR = "xor"      # Bitwise XOR
    RET = "ret"      # Return from procedure
    CMPQ = "cmpq"    # Compare quadword
    INCQ = "incq"    # Increment quadword
    JE = "je"        # Jump if equal
    JNE = "jne"      # Jump if not equal
    JNL = "jnl"      # Jump if not less
    JNLE = "jnle"    # Jump if not less or equal
    JNG = "jng"      # Jump if not greater
    JNGE = "jnge"    # Jump if not greater or equal
    JZ = "jz"        # Jump if zero
    JMP = "jmp"      # Unconditional jump
    CALL = "call"    # Call procedure
    TEST = "test"    # Logical compare
    MOV = "mov"      # Move data
    CMP = "cmp"      # Compare

    def Emit(self) -> str:
        return self.value


class Instruction(Emitter):
    """
    Represents a single assembly instruction composed of an opcode and optional arguments.

    Attributes:
        opcode (OpCode): The operation code for the instruction.
        operands (List[Emitter]): List of operands
    """

    def __init__(self, opcode: OpCode, operands: List[Emitter] = []) -> None:
        self.opcode: OpCode = opcode
        self.operands: List[Emitter] = operands

    def Emit(self) -> str:
        return "\t{}\t{}".format(self.opcode.Emit(), ", ".join(arg.Emit() for arg in self.operands))


class Directive(Emitter, Enum):
    """
    Enum for x86-64 assembler directives.
    """

    SectionText = ".section .text"   # Start of code section
    SectionData = ".section .data"   # Start of data section
    GlobalMain = ".globl main"       # Make 'main' global

    def Emit(self) -> str:
        return self.value


class Assembler(object):
    def __init__(self, output_file: TextIO = sys.stdout) -> None:
        """
        Initialize the Assembler.

        Args:
            output_file (TextIO): The output stream for emitting assembly code.
        """
        self.output_file: TextIO = output_file
        # Current batch of instructions, flushed on label and end of function
        self.batch: List[Instruction] = []

    def flush(self) -> None:
        """
        Emit all pending instructions in the batch to the output.
        Clears the batch after emitting.
        """
        for instruction in self.batch:
            print(instruction.Emit(), file=self.output_file)
        self.batch = []

    def directive(self, directive: Directive) -> None:
        """
        Emit an assembly directive.

        Args:
            directive (Directive): The directive to emit.
        """
        self.flush()
        print(directive.Emit(), file=self.output_file)

    def comment(self, line: str = "") -> None:
        """
        Emit a comment in the assembly output.

        Args:
            line (str): The comment text (default is an empty line).
        """
        self.flush()
        print("# {comment}".format(comment=line), file=self.output_file)

    def label(self, label: str):
        """
        Emit a label in the assembly output.

        Args:
            label (str): The name of the label to emit.
        """
        self.flush()
        print('{}:'.format(label), file=self.output_file)

    def instruction(self, opcode: OpCode, *operands: List[Emitter]):
        self.batch.append(Instruction(opcode=opcode, operands=operands))


class Context(object):

    def __init__(self) -> None:
        self.current_function: ast.FunctionDef = None

    def setFunction(self, function: ast.FunctionDef):
        self.current_function = function

    def resetFunction(self):
        self.current_function = None

    def isInAFunction(self) -> bool:
        return self.current_function != None


class Compiler(object):

    def __init__(self):
        self.assembler: Assembler = Assembler()
        self.ctx: Context = Context()

    def header(self) -> None:
        """
        Emits the beginning of the assembly program.

        Sets the text section and inserts an optional comment.
        """
        self.assembler.directive(Directive.SectionText)
        self.assembler.comment()

    def footer(self) -> None:
        """
        Emits the end of the assembly program.

        Appends a comment indicating the end of the generated code.
        """
        self.assembler.comment("the end")

    def visit(self, node: ast.AST):
        """
        Dispatches the given AST node to the corresponding visitor method.

        Args:
            node (ast.AST): The AST node to visit.

        Raises:
            Exception: If no matching visit method is implemented for the node type.
        """
        name = node.__class__.__name__
        visit_func: Callable = getattr(self, "visit{}".format(name), None)
        if visit_func == None:
            raise Exception(
                "'{} not supported - node {}'".format(name, ast.dump(node)))
        visit_func(node)

    def visitModule(self, node: ast.Module):
        """
        Visit all statements in the module body.

        Args:
            node (ast.Module): The module AST node to visit.
        """
        for statement in node.body:
            self.visit(statement)

    def visitFunctionDef(self, node: ast.FunctionDef):
        assert self.ctx.isInAFunction() is not None, 'nested functions not supported'
        assert node.args.vararg is None, '*args not supported'
        assert not node.args.kwonlyargs, 'keyword-only args not supported'
        assert not node.args.kwarg, 'keyword args not supported'

        def functionHeader():
            """
            Emit the function label
            """
            if node.name == "main":
                self.assembler.directive(Directive.GlobalMain)
            self.assembler.label(node.name)

        def updateStack():
            """
            Emit the standard function prologue to set up a new stack frame.

            This includes:
            - Saving the previous base pointer (%rbp)
            - Setting the new base pointer to the current stack pointer (%rsp)
            """
            # Save the caller's base pointer onto the stack
            self.assembler.instruction(OpCode.PUSHQ, Register.RBP)

            # Set the new base pointer to the current stack pointer
            self.assembler.instruction(OpCode.MOVQ, Register.RSP, Register.RBP)

        def allocateStackSpace():
            pass

        def visitBody():
            """
            Visit and compile each statement in the function body.
            """
            for statement in node.body:
                self.visit(statement)

        def functionFooter():
            """
            Emit the function epilogue: ensures return sequence exists, 
            restores stack frame, and exits cleanly if necessary.
            """
            if not isinstance(node.body[-1], ast.Return):
                # If the last statement is not an explicit return...
                self.visitReturn(ast.Return())

        self.ctx.setFunction(node)        # Set the current function context
        functionHeader()                  # Emit function-level directives and label
        updateStack()                     # Set up stack frame (push rbp, mov rsp to rbp)
        allocateStackSpace()
        visitBody()                       # Compile the function body
        functionFooter()                  # Tear down stack frame and return
        self.ctx.resetFunction()          # Clear the function context after codegen

    def visitPass(self, node: ast.Pass):
        pass

    def visitReturn(self, node: ast.Return):
        """
        Handle a return statement by compiling the return value (if any),
        and moving it into RAX before returning from the function.

        Args:
            node (ast.Return): The return statement node.
        """
        if node.value:
            # Compile the expression to evaluate the return value
            self.visit(node.value)
            # Pop return value into RAX (TODO: test function calls here)
            self.assembler.instruction(OpCode.POPQ, Register.RAX)

        # If main has no explicit return, return 0
        if self.ctx.current_function.name == 'main' and not node.value:
            self.assembler.instruction(OpCode.MOVQ, Literal(0), Register.RAX)

        # Restore previous base pointer (stack frame)
        self.assembler.instruction(OpCode.POPQ, Register.RBP)

        # Return from function
        self.assembler.instruction(OpCode.RET)

    def visitConstant(self, node: ast.Constant):
        """
        Handle an integer constant by pushing it to the stack.

        Args:
            node (ast.Constant): The constant node in the AST.

        Raises:
            AssertionError: If the constant is not an integer.
        """
        if isinstance(node.value, int):
            # Push integer constant to stack
            self.assembler.instruction(OpCode.PUSHQ, Literal(node.n))
        else:
            assert False, 'only int constants supported'

    def compile(self, node: ast.Module):
        """
        Compile the given AST module into assembly output.

        Args:
            node (ast.Module): The root node of the abstract syntax tree to compile.
        """
        self.header()
        self.visit(node)
        self.footer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='filename to compile')
    args = parser.parse_args()

    with open(args.filename) as f:
        source = f.read()
    node = ast.parse(source, filename=args.filename)
    compiler = Compiler()
    compiler.compile(node)
