/**
This benchmark tests parallelizing the sorting of large arrays of strings
using a fairly trivial modification of the quick sort algorithm.  At each
recursion in the parallel case we use a Task object to do the non-tail
recursive call instead of simply executing it in serial.
*/

import std.stdio, std.datetime, std.parallelism, std.random, std.exception,
    std.algorithm, std.getopt;

// Minimum size of subproblem to parallelize.
enum minParallel = 64;

/**
This enum decides whether our quick sort gets parallelized or not.
*/
enum Parallel : bool {
    yes = true,
    no = false
}

bool comp(T)(T a, T b) { return a < b; }

void qsort(Parallel parallel, T)(T[] data) {
    if(data.length < 25) {
         insertionSort(data);
         return;
    }

    // Just use the middle element as the pivot since this is a proof of
    // concept that will always be tested on random, non-pathological data.
    {
        immutable size_t mid = data.length / 2;
        auto temp = data[mid];
        data[mid] = data[$ - 1];
        data[$ - 1] = temp;
    }

    T[] less, greater;
    size_t lessI = size_t.max, greaterI = data.length - 1;

    auto pivot = data[$ - 1];

    while(true) {
        while(comp(data[++lessI], pivot)) {}
        while(greaterI > 0 && comp(pivot, data[--greaterI])) {}

        if(lessI < greaterI) {
            auto temp = data[lessI];
            data[lessI] = data[greaterI];
            data[greaterI] = temp;
        } else break;
    }

    auto temp = data[$ - 1];
    data[$ - 1] = data[lessI];
    data[lessI] = temp;
    less = data[0..min(lessI, greaterI + 1)];
    greater = data[lessI + 1..$];

    Task!(qsort!(Parallel.yes, T), T[]) recurseTask;
    if(greater.length > less.length) {
        if(parallel && less.length >= minParallel) {
            recurseTask = scopedTask!(qsort!(Parallel.yes, T))(less);
            taskPool.put(recurseTask);
        } else {
            qsort!(parallel)(less);
        }
        qsort!(parallel)(greater);
    } else {
        if(parallel && greater.length >= minParallel) {
            recurseTask = scopedTask!(qsort!(Parallel.yes, T))(greater);
            taskPool.put(recurseTask);
        } else {
            qsort!(parallel, T)(greater);
        }
        qsort!(parallel, T)(less);
    }

    // Relying on the destructor of stack-allocated Task objects to wait for
    // recurseTask to be finished.
}

T[] insertionSort(T)(T[] data) {
    if(data.length < 2) {
        return data;
    }

    // Yes, I measured this, caching this value is actually faster on DMD.
    immutable maxJ = data.length - 1;
    for(size_t i = data.length - 2; i != size_t.max; --i) {
        size_t j = i;

        T temp = data[i];
        for(; j < maxJ && comp(data[j + 1], temp); ++j) {
            data[j] = data[j + 1];
        }

        data[j] = temp;
    }

    return data;
}

string randString() {
    auto ret = new char[uniform(20, 50)];
    foreach(ref c; ret) {
        c = cast(char) uniform('a', 'z' + 1);
    }

    return assumeUnique(ret);
}

enum nIter = 1_000;

void main(string[] args) {
    uint nCpu = uint.max;
    getopt(args, "nCpu", &nCpu);
    if(nCpu < uint.max && nCpu > 0) {
        defaultPoolThreads = nCpu - 1;
    }

    auto strings = new string[10_000];
    foreach(ref s; strings) s = randString();
    auto strings2 = strings.dup;

    auto sw = StopWatch(autoStart);

    // Test the serial case first.
    foreach(iter; 0..nIter) {
        randomShuffle(strings);
        qsort!(Parallel.no)(strings);
    }

    writefln("Serial quick sort:  %d milliseconds.", sw.peek.msecs);

    sw.reset();

    // Parallel case.  The shuffling appears to take a substantial amount of
    // time, too, so parallelize that with the sorting for the purposes
    // of this benchmark.
    foreach(iter; 0..nIter) {
        swap(strings, strings2);
        auto shuffleTask = task!randomShuffle(strings2);
        taskPool.put(shuffleTask);
        qsort!(Parallel.yes)(strings);
    }

    writefln("Parallel quick sort:  %d milliseconds.", sw.peek.msecs);
}
