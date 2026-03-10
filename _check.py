import py_compile, sys
try:
    py_compile.compile(r'C:\Bot\cryptotrades\core\trading_engine.py', doraise=True)
    print("COMPILE_OK")
    sys.exit(0)
except py_compile.PyCompileError as e:
    print(f"COMPILE_FAIL: {e}")
    sys.exit(1)
