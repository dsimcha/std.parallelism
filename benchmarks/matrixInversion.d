/**
This benchmark tests the speed of inverting a large matrix using
Gauss-Jordan elimination.  Again, we do everything in single precision to
avoid stack alignment artifacts on 32-bit.

The implementation of Gauss-Jordan used here is admittedly fairly naive,
since the goal is to test the relative speedup from parallelizing using
fairly simple, readable code, not to write the fastest matrix inversion
routine humanly possible.
*/
import std.algorithm, std.stdio, std.math, std.datetime, std.parallelism,
    std.range, std.random, std.getopt, core.memory;

enum Parallel : bool {
    yes = true,
    no = false
}

// n by n matrix.
enum n = 1024;

void invert(Parallel parallel)(float[][] from, float[][] to) {
    // Normalize.  This is probably not worth parallelizing.
    foreach(i, row; from) {
        float absMax = 1.0 / reduce!(max)(map!(std.math.abs)(row));
        row[] *= absMax;
        to[i][] = 0;
        to[i][i] = absMax;
    }

    foreach(col; 0..from.length) {
        size_t bestRow;
        float biggest = 0;
        foreach(row; col..from.length) {
            if(abs(from[row][col]) > biggest) {
                bestRow = row;
                biggest = abs(from[row][col]);
            }
        }

        swap(from[col], from[bestRow]);
        swap(to[col], to[bestRow]);
        immutable pivotFactor = from[col][col];

        static if(parallel) {
            auto rowIter = taskPool.parallel(iota(from.length));
        } else {
            auto rowIter = iota(from.length);
        }

        // This block is by far the biggest bottleneck in matrix inversion,
        // thus it's the only one place parallelism is used in this algorithm.
        foreach(row; rowIter) if(row != col) {
            immutable ratio = from[row][col] / pivotFactor;

            from[row][] -= from[col][] * ratio;
            to[row][] -= to[col][] * ratio;
        }
    }

    foreach(i; 0..from.length) {
        immutable diagVal = from[i][i];
        from[i][] /= diagVal;
        to[i][] /= diagVal;
    }
}

// Generate a random n by n matrix.
float[][] randMatrix() {
    Random gen = Random(867_5309);  // Random but repeatable.
    auto ret = new float[][](n, n);

    foreach(row; ret) foreach(ref elem; row) {
        elem = uniform(-1.0, 1.0);
    }

    return ret;
}

void main(string[] args) {
    uint nCpu = uint.max;
    getopt(args, "nCpu", &nCpu);
    if(nCpu < uint.max && nCpu > 0) {
        defaultPoolThreads = nCpu - 1;
    }

    auto toMatrix = new float[][](n, n);
    auto serialFrom = randMatrix();

    // Make sure initialization overhead doesn't affect our benchmark.
    taskPool();

    auto sw = StopWatch(autoStart);
    invert!(Parallel.no)(serialFrom, toMatrix);
    writefln("Inverted a %s x %s matrix serially in %s milliseconds.",
        n, n, sw.peek.msecs
    );

    auto parallelFrom = randMatrix();
    sw.reset();
    invert!(Parallel.yes)(parallelFrom, toMatrix);
    writefln("Inverted a %s x %s matrix using %s cores in %s milliseconds.",
        n, n, taskPool.size() + 1, sw.peek.msecs
    );
}
