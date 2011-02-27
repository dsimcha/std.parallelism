/**
This is a benchmark of calculating pi, both serially and using parallelism,
using the formula pi = 4/1 - 4/3 + 4/5 - 4/7 + 4/9 - ...

Note that the serial and parallel algorithms will give slightly different
answers because, in inexact floating point arithmetic, addition is not
associative.
*/

import std.stdio, std.datetime, std.parallelism, std.range, std.algorithm,
    std.getopt;

float getTerm(float i) pure nothrow {
    auto term = 4.0f / i;
    return ((i - 1) % 4) ? -term : term;
}

enum float n = 10_000_000;

void main(string[] args) {
    uint nCpu = uint.max;
    getopt(args, "nCpu", &nCpu);
    if(nCpu < uint.max && nCpu > 0) {
        defaultPoolThreads = nCpu - 1;
    }

    // Make sure initialization overhead doesn't affect our benchmark.
    taskPool();

    auto sw = StopWatch(autoStart);
    immutable piSerial = std.algorithm.reduce!"a + b"(
        std.algorithm.map!getTerm(
            iota(1f, n, 2f)
        )
    );
    writefln("Calculated pi = %.10s in %s milliseconds serially.",
        piSerial, sw.peek.msecs);

    sw.reset();
    immutable piParallel = taskPool.reduce!"a + b"(
        std.algorithm.map!getTerm(
            iota(1f, n, 2f)
        )
    );
    writefln("Calculated pi = %.10s in %s milliseconds using %s cores.",
        piParallel, sw.peek.msecs, taskPool.size + 1);
}
