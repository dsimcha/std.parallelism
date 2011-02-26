/**
This is a very micro benchmark of parallel reduce, and tests the speed with
which the sum of squares of a large array can be found.  Everything is
done in single precision to avoid artifacts due to stack alignment on 32-bit
architectures.
*/

import std.stdio, std.algorithm, std.parallelism, std.random, std.datetime;

enum nIter = 1000;

void main() {
    auto arr = new float[250_000];
    foreach(ref elem; arr) elem = uniform(-0.5, 0.5);

    // Make sure the task pool is initialized upfront so any one-time
    // initialization overhead doesn't factor into our benchmark.
    taskPool();

    auto sw = StopWatch(autoStart);
    foreach(iter; 0..nIter) {
        reduce!"a + b * b"(arr);
    }
    writefln("Serial reduce:  %s milliseconds.", sw.peek.msecs);

    sw.reset();
    foreach(iter; 0..nIter) {
        taskPool.reduce!"a + b * b"(arr);
    }
    writefln("Parallel reduce with %s cores:  %s milliseconds.",
        taskPool.size() + 1, sw.peek.msecs);
}

