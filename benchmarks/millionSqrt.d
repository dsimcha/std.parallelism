/**
Test parallel map and parallel foreach's ability to speed up calculation
of the sqrt of a million element array.  Parallel foreach is slightly
slower here because it uses a delegate call on each iteration.  However,
this increases flexibility and the penalty is very small.  Parallel map
appears to give a linear speedup here.
*/

import std.stdio, std.datetime, std.parallelism, std.math, std.algorithm,
    std.range, std.array, std.getopt;

enum nIter = 100;

void main(string[] args) {
    uint nCpu = uint.max;
    getopt(args, "nCpu", &nCpu);
    if(nCpu < uint.max && nCpu > 0) {
        defaultPoolThreads = nCpu - 1;
    }

    immutable nCores = taskPool.size() + 1;
    writefln("Parallel benchmarks being done with %s cores.", nCores);

    auto nums = array(iota(1_000_000.0f));
    auto buf = new float[nums.length];

    // Test serial first.
    auto sw = StopWatch(autoStart);

    foreach(iter; 0..nIter) {
        foreach(i, num; nums) {
            buf[i] = sqrt(num);
        }
    }
    writefln("Did serial millionSqrt in %s milliseconds.", sw.peek.msecs);

    sw.reset();
    foreach(iter; 0..nIter) {
        foreach(i, num; nums.parallel()) {
            buf[i] = sqrt(num);
        }
    }
    writefln("Did parallel foreach millionSqrt in %s milliseconds.",
        sw.peek.msecs);

    sw.reset();
    foreach(iter; 0..nIter) {
        taskPool.map!sqrt(nums, buf);
    }
    writefln("Did parallel map millionSqrt in %s milliseconds.", sw.peek.msecs);
}
