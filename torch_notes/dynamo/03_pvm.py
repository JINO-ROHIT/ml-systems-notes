import dis

def foo(a, b):
    return a + b

dis.dis(foo)