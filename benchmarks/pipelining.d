/**
This benchmark tests the efficiency of pipelining using AsyncBuf and
LazyMap.  We first create an input range for generating a large number of
numeric strings.  We then use LazyMap and AsyncBuf to pipeline reading from
this range, converting its values from strings to floats, rounding these
floats to ints, and finding the greatest common divisor of these ints with
the nunmber 8675309.
*/
import std.stdio, std.datetime, std.parallelism, std.random, std.algorithm,
    std.range, std.numeric, std.conv, std.getopt;

enum nTake = 1_000_000;

/**This is an input range of random numeric strings.*/
struct StringNums {
    Random gen;
    string _front;
    float maxVal;

    this(float maxVal) {
        this.maxVal = maxVal;
        this.gen = Random(31415);  // Random but repeatable.
        popFront();
    }

    string front() @property { return _front; }

    void popFront() {
        _front = to!string(uniform(0f, maxVal));
    }

    enum bool empty = false;
}


void main(string[] args) {
    uint nCpu = uint.max;
    getopt(args, "nCpu", &nCpu);
    if(nCpu < uint.max && nCpu > 0) {
        defaultPoolThreads = nCpu - 1;
    }

    auto sw = StopWatch(autoStart);
    doSerial();
    writefln("Did serial string -> float, euclid in %s milliseconds.",
        sw.peek.msecs);

    sw.reset();
    doParallel();
    writefln("Did parallel string -> float, euclid with %s cores in %s milliseconds.",
        taskPool.size() + 1, sw.peek.msecs);
}

int gcd8675309(int num) {
    return gcd(num, 8675309);
}

void doSerial() {
    auto numStrs = StringNums(8675309);
    foreach(numStr; take(numStrs, nTake)) {
        auto f = to!float(numStr);
        auto i = roundTo!int(f);
        auto divisor = gcd8675309(i);
    }
}

void doParallel() {
    auto numStrs = StringNums(8675309);
    auto numStrBuf = taskPool.asyncBuf(take(numStrs, nTake), 1024);
    auto floats = taskPool.lazyMap!(to!float)(numStrBuf);
    auto ints = taskPool.lazyMap!(to!int)(floats);

    auto gcds = taskPool.lazyMap!gcd8675309(ints);
    foreach(elem; gcds) {}
}
