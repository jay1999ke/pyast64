import argparse
import ast
from enum import Enum
import sys
from typing import Callable, Dict, List, TextIO
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

    @staticmethod
    def functionOrder() -> List:
        return [
            Register.RDI,
            Register.RSI,
            Register.RDX,
            Register.RCX,
            Register.R8,
            Register.R9]


class Literal(Emitter):

    def __init__(self, value: int) -> None:
        self.value: int = value

    def Emit(self) -> str:
        return "${}".format(self.value)


class LiteralString(Emitter):

    def __init__(self, index: int, value: str) -> None:
        self.index: int = index
        self.value: str = value

    def Emit(self) -> str:
        return "$str{}".format(self.index)

    def getLabel(self) -> str:
        return "str{}".format(self.index)


class Label(Emitter):

    def __init__(self, name: str) -> None:
        self.name: str = name

    def Emit(self) -> str:
        return self.name


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
    SUB = "sub"      # Subtract word
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
    String = "place_holder"

    def Emit(self) -> str:
        if self == Directive.String:
            escaped: str = self.string.encode('unicode_escape').decode('ascii')
            return '.asciz \"{}\"'.format(escaped)
        return self.value

    def AsString(self, string: str) -> Emitter:
        self.string = string
        return self


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
    """
    Holds the current state of code generation, such as the current function,
    local variable mappings, and function labels.
    """

    def __init__(self) -> None:
        # The function currently being compiled
        self.current_function: ast.FunctionDef = None

        # All declared function labels (e.g., function names)
        self.labels: Dict[str, Label] = {}
        self.libc_labels: Dict[str, Label] = {
            "printf": Label("printf"),
            "putchar": Label("putchar"),
            "puts": Label("puts"),
            "print": Label("printf"), # Make it pretty for python 3 linters, code-formatters
        }

        # Maps local variable names to their index (stack slot)
        self.locals: Dict[str, int] = {}

        # Counter for number of local variables seen so far
        self.locals_index = 0

        # set allocated space
        self.allocated_space_size = 0

        # for unique labels
        self.label_num = 1

        # static strings
        self.strings: Dict[str, LiteralString] = {}

        # break labels stack
        self.break_labels: List[Label] = []

    def label(self, slug: str) -> str:
        label = '{}_{}_{}'.format(
            self.current_function.name, self.label_num, slug)
        self.label_num += 1
        return label

    def setAllocatedSpace(self, space: int):
        self.allocated_space_size = space

    def setFunction(self, function: ast.FunctionDef):
        """
        Set the function currently being compiled.
        """
        self.current_function = function

    def resetFunction(self):
        """
        Reset the function context after function compilation is complete.
        Clears locals and resets index.
        """
        self.current_function = None
        self.locals = {}
        self.locals_index = 0
        self.allocated_space_size = 0
        self.label_num = 1
        self.break_labels = []

    def addLocal(self, local: str):
        """
        Register a new local variable by name.

        Args:
            local (str): The variable name to track.
        """
        self.locals[local] = self.locals_index
        self.locals_index += 1

    def bumpLocalIndexForRIP(self):
        self.locals_index += 1

    def isALocal(self, string: str) -> bool:
        return string in self.locals

    def getLocalOffset(self, local_name: str) -> int:
        """
        Compute the stack offset for a given local variable.

        Args:
            local_name (str): The name of the local variable.

        Returns:
            int: The offset from %rbp to access the variable.
        """
        index: int = self.locals[local_name]

        # Earlier locals are deeper in the stack
        offset: int = len(self.locals) - index

        # Account for return address pushed by 'call'
        offset += 1

        return offset * 8

    def isInAFunction(self) -> bool:
        """
        Check if we are currently inside a function.

        Returns:
            bool: True if in a function context, else False.
        """
        return self.current_function is not None

    def newLabel(self, label: Label):
        """
        Register a new function label.

        Args:
            label (Label): The label to add.

        Raises:
            Exception: If the label is already defined.
        """
        if label.name in self.labels:
            raise Exception("Redefinition of function '{}'".format(label.name))
        self.labels[label.name] = label

    def break_label_stack_push(self, label: Label):
        self.break_labels.append(label)

    def break_label_stack_tos(self) -> Label:
        return self.break_labels[-1]

    def break_label_stack_pop(self):
        self.break_labels.pop()

    def getLabel(self, label: str, raise_exception: bool = True) -> Label:
        """
        Retrieve a previously registered label.

        Args:
            label (str): The name of the label.

        Returns:
            Label: The corresponding Label object.

        Raises:
            Exception: If the label is not defined.
        """
        if label not in self.labels and label not in self.libc_labels:
            if raise_exception:
                raise Exception(
                    "Calling an unknown function '{}'".format(label))
            else:
                return None
        if label in self.libc_labels:
            return self.libc_labels[label]
        return self.labels[label]

    def getLibcLabel(self, label: str, raise_exception: bool = True) -> Label:
        """
        Retrieve a previously registered label.

        Args:
            label (str): The name of the label.

        Returns:
            Label: The corresponding Label object.

        Raises:
            Exception: If the label is not defined.
        """
        if label not in self.libc_labels:
            if raise_exception:
                raise Exception(
                    "Calling an unknown function '{}'".format(label))
            else:
                return None
        return self.libc_labels[label]


class LocalsVisitor(ast.NodeVisitor):
    """
    Recursively visit a FunctionDef node to find all local variables.

    This helps determine how much stack space is needed by collecting
    all variable names assigned within the function, excluding those
    marked as global.
    """

    def __init__(self):
        self.local_names: List[str] = []   # List of local variable names
        self.global_names: List[str] = []  # List of global variable names

    def add(self, name: str):
        """
        Add a variable name to the list of locals, unless it is declared global.

        Args:
            name (str): The variable name to add.
        """
        if name not in self.local_names and name not in self.global_names:
            self.local_names.append(name)

    def visit_Global(self, node: ast.Global):
        """
        Handle `global` statements to prevent those names from being treated as locals.

        Args:
            node (ast.Global): The global statement AST node.
        """
        self.global_names.extend(node.names)  # Mark these names as global

    def visit_Assign(self, node):
        """
        Visit an assignment statement and collect the target variable name.

        Args:
            node (ast.Assign): The assignment AST node.

        Raises:
            AssertionError: If the assignment has multiple targets.
        """
        assert len(node.targets) == 1, \
            'can only assign one variable at a time'  # Only handle single assignment
        self.visit(node.value)  # Visit the value to catch nested assignments
        target = node.targets[0]
        self.add(target.id)  # Add the assigned variable to locals

    def visit_For(self, node: ast.For):
        """
        Visit a `for` loop and register the loop variable as a local.

        Args:
            node (ast.For): The for-loop AST node.
        """
        self.add(node.target.id)  # Add loop variable as a local
        for statement in node.body:
            self.visit(statement)  # Visit loop body for nested locals


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
        if len(self.ctx.strings) > 0:
            self.assembler.directive(Directive.SectionData)
            for string in self.ctx.strings.keys():
                literal: LiteralString = self.ctx.strings[string]
                self.assembler.label(literal.getLabel())
                self.assembler.directive(Directive.String.AsString(string))
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

        def fetchArguments():
            """
            Add arguments to local variables
            """
            for arg in node.args.args:
                self.ctx.addLocal(arg.arg)

        def functionHeader():
            """
            Emit the function label
            """
            if node.name == "main":
                self.assembler.directive(Directive.GlobalMain)
            self.ctx.newLabel(Label(node.name))
            self.assembler.label(node.name)

        def allocateStackSpace() -> int:
            """
            Allocate stack space for all local variables used in the function,
            including those assigned dynamically. This is done before setting up
            the stack frame so that locals and function arguments are treated uniformly.
            """

            # Find names of additional locals assigned in this function
            locals_visitor = LocalsVisitor()
            locals_visitor.visit(node)

            # If there are local variables used in this function
            if len(locals_visitor.local_names) > 0:
                # Add a dummy local to simulate space taken by return address (RIP)
                self.ctx.bumpLocalIndexForRIP()

                # Add new local variables to the context if not already present
                for name in locals_visitor.local_names:
                    if not self.ctx.isALocal(name):
                        self.ctx.addLocal(name)

            # Calculate how many additional local slots to allocate (excluding arguments)
            num_extra_locals = len(self.ctx.locals) - len(node.args.args)

            if num_extra_locals > 0:

                # Allocate space for locals in one go using subq
                # Each local is 8 bytes (quadword)
                total_bytes = num_extra_locals * 8
                self.assembler.instruction(
                    OpCode.SUBQ, Literal(total_bytes), Register.RSP)
                return num_extra_locals
            return 0

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

        # Set the current function context
        self.ctx.setFunction(node)

        # Gather our arguments
        fetchArguments()

        # Emit function-level directives and label
        functionHeader()

        # Allocate space for local variables before prologue
        # so locals and function args are handled uniformly
        self.ctx.setAllocatedSpace(allocateStackSpace())

        # Set up stack frame (push rbp, mov rsp to rbp)
        updateStack()

        # Compile the function body
        visitBody()

        # Tear down stack frame and return
        functionFooter()

        # Clear the function context after codegen
        self.ctx.resetFunction()

        # visibility
        self.assembler.comment()

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

        # Deallocate stack space
        if self.ctx.allocated_space_size > 0:
            offset: int = self.ctx.allocated_space_size * 8
            self.assembler.instruction(
                OpCode.ADDQ, Literal(offset), Register.RSP)

        # Return from function
        self.assembler.instruction(OpCode.RET)

    def visitCall(self, node: ast.Call):
        """
        Compile a function call expression.

        Args:
            node (ast.Call): The AST node representing a function call.
        """

        def libCFuncsSetup():
            if self.ctx.getLibcLabel(node.func.id, False):
                assert len(node.args) < 7, \
                    'can only call C lib funcs with 6 or less args'
                for idx in range(len(node.args), 0, -1):
                    self.assembler.instruction(
                        OpCode.POPQ, Register.functionOrder()[idx-1])
                # stack align to 16 bytes for certain libc functions
                if (node.func.id.endswith('printf')):
                    label: Label = Label(self.ctx.label('skip_push'))
                    self.ctx.newLabel(label)
                    self.assembler.instruction(
                        OpCode.TEST, Literal(8), Register.RSP)
                    self.assembler.instruction(OpCode.JE, label)
                    self.assembler.instruction(OpCode.PUSHQ, Register.RSP)
                    self.assembler.label(label.Emit())

        def libCFuncsTeardown():
            if self.ctx.getLibcLabel(node.func.id, False):
                # libc alignment cleanup
                if (node.func.id.endswith('printf')):
                    label: Label = Label(self.ctx.label('skip_pop'))
                    self.ctx.newLabel(label)
                    self.assembler.instruction(
                        OpCode.MOV, Dereference(Register.RSP).WithOffset(0), Register.RDX)
                    self.assembler.instruction(
                        OpCode.SUB, Literal(8), Register.RDX)
                    self.assembler.instruction(
                        OpCode.CMP, Register.RDX, Register.RSP)
                    self.assembler.instruction(OpCode.JNE, label)
                    self.assembler.instruction(OpCode.POPQ, Register.RDX)
                    self.assembler.label(label.Emit())

        # Evaluate each argument and push it on the stack (right-to-left order)
        for arg in node.args:
            self.visit(arg)

        # special setup for libc functions
        libCFuncsSetup()

        # Call the function by label (function name must be known)
        self.assembler.instruction(
            OpCode.CALL, self.ctx.getLabel(node.func.id))

        # special teardown for libc functions
        libCFuncsTeardown()

        # After call, adjust the stack to clean up arguments
        if self.ctx.getLabel(node.func.id) and not self.ctx.getLibcLabel(node.func.id, False) and node.args:
            self.assembler.instruction(
                OpCode.ADDQ, Literal(8 * len(node.args)), Register.RSP)

        # Function return value is in %rax — push it on the stack for later use
        self.assembler.instruction(OpCode.PUSHQ, Register.RAX)

    def visitName(self, node: ast.Name):
        """
        Compile a reference to a local variable.

        Args:
            node (ast.Name): The AST node representing a variable name.
        """
        # Lookup local variable offset and push its value from the stack
        offset = self.ctx.getLocalOffset(node.id)
        self.assembler.instruction(
            OpCode.PUSHQ, Dereference(Register.RBP).WithOffset(offset))

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
        elif isinstance(node.value, str):
            if not node.value in self.ctx.strings:
                self.ctx.strings[node.value] = LiteralString(
                    len(self.ctx.strings), node.value)
            self.assembler.instruction(
                OpCode.PUSHQ, self.ctx.strings[node.value])
        else:
            assert False, 'only int constants supported'

    def visitAssign(self, node: ast.Assign):
        """
        Compile an assignment statement.

        This method assumes only a single local variable is being assigned.
        The right-hand side is compiled first (its value pushed to stack),
        then the value is popped into the correct stack offset.

        Args:
            node (ast.Assign): The assignment node from the AST.
        """
        # Only support single-target assignments (e.g., x = 42)
        assert len(node.targets) == 1, \
            'can only assign one variable at a time'

        # Compile the expression on the right-hand side of the assignment
        self.visit(node.value)

        # Store the result in the stack slot for the local variable
        offset = self.ctx.getLocalOffset(node.targets[0].id)
        self.assembler.instruction(
            OpCode.POPQ, Dereference(Register.RBP).WithOffset(offset))

    def visitAugAssign(self, node: ast.AugAssign):
        """
        Compile an augmented assignment statement (e.g., x += 1).

        This works by:
        1. Loading the current value of the variable.
        2. Compiling the right-hand side expression.
        3. Performing the binary operation.
        4. Storing the result back to the same variable.

        Args:
            node (ast.AugAssign): The augmented assignment AST node.
        """
        # Load current value of the target variable
        self.visit(node.target)

        # Compile right-hand side expression
        self.visit(node.value)

        # Compile the operator
        self.visit(node.op)

        # Store result back to the same stack location
        offset = self.ctx.getLocalOffset(node.target.id)
        self.assembler.instruction(
            OpCode.POPQ, Dereference(Register.RBP).WithOffset(offset))

    def visitExpr(self, node: ast.Expr):
        """
        Handle an expression statement by evaluating the expression
        and storing its result in the RAX register.

        Args:
            node (ast.Expr): An expression node.
        """
        # Evaluate the expression and push result onto the stack
        self.visit(node.value)

        # Store the result in RAX
        self.assembler.instruction(OpCode.POPQ, Register.RAX)

    #
    # Operations follow
    #

    def _simple_binop(self, op: OpCode):
        self.assembler.instruction(OpCode.POPQ, Register.RDX)
        self.assembler.instruction(OpCode.POPQ, Register.RAX)
        self.assembler.instruction(op, Register.RDX, Register.RAX)
        self.assembler.instruction(OpCode.PUSHQ, Register.RAX)

    def visitBinOp(self, node: ast.BinOp):
        self.visit(node.left)
        self.visit(node.right)
        self.visit(node.op)

    def visitAdd(self, node: ast.Add):
        self._simple_binop(OpCode.ADDQ)

    def visitMult(self, node: ast.Mult):
        self.assembler.instruction(OpCode.POPQ, Register.RDX)
        self.assembler.instruction(OpCode.POPQ, Register.RAX)
        self.assembler.instruction(OpCode.IMULQ, Register.RDX)
        self.assembler.instruction(OpCode.PUSHQ, Register.RAX)

    def visitCompare(self, node: ast.Compare):
        assert len(node.ops) == 1, 'only single comparisons supported'
        self.visit(node.left)
        self.visit(node.comparators[0])
        self.visit(node.ops[0])

    def _compile_comparison(self, jump_not: OpCode, slug: str):
        # Generate a unique label to jump to if the comparison fails
        label: Label = Label(self.ctx.label(slug))
        self.ctx.newLabel(label)

        # Pop the right-hand side and left-hand side values from the stack
        self.assembler.instruction(OpCode.POPQ, Register.RDX)  # RHS
        self.assembler.instruction(OpCode.POPQ, Register.RAX)  # LHS

        # Compare RAX and RDX (effectively: RAX - RDX)
        self.assembler.instruction(OpCode.CMPQ, Register.RDX, Register.RAX)

        # Assume comparison is false: move 0 into RAX
        self.assembler.instruction(OpCode.MOVQ, Literal(0), Register.RAX)

        # If condition not met, jump to label (skip setting result to 1)
        self.assembler.instruction(jump_not, label)

        # Condition was true: set result in RAX to 1
        self.assembler.instruction(OpCode.INCQ, Register.RAX)

        # Label target for failed comparison or to continue after success
        self.assembler.label(label.Emit())

        # Push the result (0 or 1) back onto the stack
        self.assembler.instruction(OpCode.PUSHQ, Register.RAX)

    def visitLt(self, node: ast.Lt):
        self._compile_comparison(OpCode.JNL, 'less')

    def visitLtE(self, node: ast.LtE):
        self._compile_comparison(OpCode.JNLE, 'less_or_equal')

    def visitGt(self, node: ast.Gt):
        self._compile_comparison(OpCode.JNG, 'greater')

    def visitGtE(self, node: ast.GtE):
        self._compile_comparison(OpCode.JNGE, 'greater_or_equal')

    def visitEq(self, node: ast.Eq):
        self._compile_comparison(OpCode.JNE, 'equal')

    def visitNotEq(self, node: ast.NotEq):
        self._compile_comparison(OpCode.JE, 'not_equal')

    def visitIf(self, node: ast.If):
        # Evaluate the condition expression
        self.visit(node.test)

        # Compare result to 0 (false)
        self.assembler.instruction(OpCode.POPQ, Register.RAX)
        self.assembler.instruction(OpCode.CMPQ, Literal(0), Register.RAX)

        # Labels for else and end
        label_else = Label(self.ctx.label('else'))
        label_end = Label(self.ctx.label('end'))
        self.ctx.newLabel(label_else)
        self.ctx.newLabel(label_end)

        # Jump to else block if condition is false (== 0)
        self.assembler.instruction(OpCode.JZ, label_else)

        # Emit code for the 'if' block
        for statement in node.body:
            self.visit(statement)

        # If there's an else block, jump over it after if-block
        if node.orelse:
            self.assembler.instruction(OpCode.JMP, label_end)

        # Else label
        self.assembler.label(label_else.Emit())

        # Emit code for the 'else' block
        for statement in node.orelse:
            self.visit(statement)

        # End label
        if node.orelse:
            self.assembler.label(label_end.Emit())

    def visitWhile(self, node: ast.While):
        # Create labels for loop start and break
        label_while = Label(self.ctx.label('while'))
        label_break = Label(self.ctx.label('break'))

        self.ctx.newLabel(label_while)
        self.ctx.newLabel(label_break)
        self.ctx.break_label_stack_push(label_break)

        # Start of while loop
        self.assembler.label(label_while.Emit())

        # Evaluate the loop condition
        self.visit(node.test)

        # Compare condition to 0 (false)
        self.assembler.instruction(OpCode.POPQ, Register.RAX)
        self.assembler.instruction(OpCode.CMPQ, Literal(0), Register.RAX)

        # Jump to break if condition is false
        self.assembler.instruction(OpCode.JZ, label_break)

        # Emit loop body
        for statement in node.body:
            self.visit(statement)

        # Jump back to start of loop
        self.assembler.instruction(OpCode.JMP, label_while)

        # Break label (exit point)
        self.assembler.label(label_break.Emit())

        # Pop off the label
        self.ctx.break_label_stack_pop()

    def visitBreak(self, node: ast.Break):
        # Unconditional jump to the current break label
        self.assembler.instruction(
            OpCode.JMP, self.ctx.break_label_stack_tos())

    def visitFor(self, node: ast.For):
        # Turn for+range loop into a while loop:
        #   i = start
        #   while i < stop:
        #       body
        #       i = i + step
        assert isinstance(node.iter, ast.Call) and \
            node.iter.func.id == 'range', \
            'for can only be used with range()'
        
        # Get your start, stop and step
        range_args = node.iter.args
        if len(range_args) == 1:
            start = ast.Constant(value=0)
            stop = range_args[0]
            step = ast.Constant(value=1)
        elif len(range_args) == 2:
            start, stop = range_args
            step = ast.Constant(value=1)
        else:
            start, stop, step = range_args
            if (isinstance(step, ast.UnaryOp) and
                    isinstance(step.op, ast.USub) and
                    isinstance(step.operand, ast.Constant)):
                # Handle negative step
                step = ast.Constant(value=-step.operand.value)
            assert isinstance(step, ast.Constant) and step.value != 0, \
                'range() step must be a nonzero integer constant'
            
        # i = start
        self.visit(ast.Assign(targets=[node.target], value=start))

        # i < stop
        test = ast.Compare(
            left=node.target,
            ops=[ast.Lt() if step.value > 0 else ast.Gt()],
            comparators=[stop],
        )

        # i = i + step
        incr = ast.Assign(
            targets=[node.target],
            value=ast.BinOp(left=node.target, op=ast.Add(), right=step),
        )

        # Test, body and step
        self.visit(ast.While(test=test, body=node.body + [incr]))

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
