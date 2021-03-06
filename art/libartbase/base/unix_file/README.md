A simple C++ wrapper for Unix file I/O.

This is intended to be lightweight and easy to use, similar to Java's
RandomAccessFile and related classes. The usual C++ idioms of RAII and "you
don't pay for what you don't use" apply.

In particular, the basic RandomAccessFile interface is kept small and simple so
it's trivial to add new implementations.

This code will not log, because it can't know whether that's appropriate in
your application.

This code will, in general, return -errno on failure. If an operation consisted
of multiple sub-operations, it will return the errno corresponding to the most
relevant operation.