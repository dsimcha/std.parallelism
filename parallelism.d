/**
$(D std.parallelism) is a module that implements some high-level primitives
for shared memory SMP parallelism.  These include parallel foreach, parallel
reduce, parallel eager map, pipelining and future/promise parallelism primitives.

This module is geared towards parallelism, not concurrency.  In particular,
the default behavior on single-core machines is to use no multithreading at
all, since there are no opportunities for parallelism on such machines.

$(D std.parallelism) is based on the concept of a $(D Task).  A $(D Task) is an
object that represents the fundamental unit of work in this library and may be
executed in parallel with any other $(D Task).  Using the $(D Task) object
directly allows programming with a future/promise paradigm.  All other
supported parallelism paradigms (parallel foreach, map, reduce, pipelining)
represent an additional level of abstraction over $(D Task) in that they
automatically create one or more $(D Task) objects, or closely related types
that are conceptually identical but not part of the public API.

After creation, $(D Task) objects are submitted to a $(D TaskPool) for execution.
A $(D TaskPool) encapsulates a task queue and its associated worker threads.
A task queue is a simple FIFO queue of $(D Task) objects that have been
submitted to the $(D TaskPool) and are awaiting execution.  A worker thread is
a thread that is associated with exactly one task queue.  It executes the
$(D Task) at the front of its task queue when the task queue has work available,
or sleeps when no work is available.  Each task queue, in turn, is associated
with zero or more worker threads.  If the results of a $(D Task) are needed in
the submitting thread before execution by a worker thread has begun, the
$(D Task) can be removed from the task queue and executed immediately in the
submitting thread rather than in a worker thread.

Warning:  Most of this module completely subverts D's type system to achieve
          unchecked data sharing and cannot be used with SafeD.
          If you're looking for D's flagship message passing concurrency
          model, which can be used with SafeD, you should use
          $(D std.concurrency) instead.  However, the one exception is that
          tasks can be used safely (i.e. from SafeD) under a limited set of
          circumstances, detailed in the documentation for $(D task).

Author:  David Simcha
Copyright:  Copyright (c) 2009-2011, David Simcha.
License:    $(WEB boost.org/LICENSE_1_0.txt, Boost License 1.0)
*/
module std.parallelism;

import core.thread, core.cpuid, std.algorithm, std.range, std.c.stdlib, std.stdio,
    std.exception, std.functional, std.conv, std.math, core.memory, std.traits,
    std.typetuple, core.stdc.string, std.typecons;

import core.sync.condition, core.sync.mutex, core.atomic;

// Workaround for bug 3753.
version(Posix) {
    // Can't use alloca() because it can't be used with exception handling.
    // Use the GC instead even though it's slightly less efficient.
    void* alloca(size_t nBytes) {
        return GC.malloc(nBytes);
    }
} else {
    // Can really use alloca().
    import core.stdc.stdlib : alloca;
}

version(Windows) {
    // BUGS:  Only works on Windows 2000 and above.

    import core.sys.windows.windows;

    struct SYSTEM_INFO {
      union {
        DWORD  dwOemId;
        struct {
          WORD wProcessorArchitecture;
          WORD wReserved;
        };
      };
      DWORD     dwPageSize;
      LPVOID    lpMinimumApplicationAddress;
      LPVOID    lpMaximumApplicationAddress;
      LPVOID    dwActiveProcessorMask;
      DWORD     dwNumberOfProcessors;
      DWORD     dwProcessorType;
      DWORD     dwAllocationGranularity;
      WORD      wProcessorLevel;
      WORD      wProcessorRevision;
    }

    private extern(Windows) void GetSystemInfo(void*);

    shared static this() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        osReportedNcpu = max(1, cast(uint) si.dwNumberOfProcessors);
    }

} else version(Posix) {
    import core.sys.posix.unistd;

    shared static this() {
        osReportedNcpu = cast(uint) sysconf(_SC_NPROCESSORS_ONLN );
    }
} else {
    static assert(0, "Don't know how to get N CPUs on this OS.");
}

/* Atomics code.  These just forward to core.atomic, but are written like this
   for two reasons:

   1.  They used to actually contain ASM code and I don' want to have to change
       to directly calling core.atomic in a zillion different places.

   2.  core.atomic has some misc. issues that make my use cases difficult
       without wrapping it.  If I didn't wrap it, casts would be required
       basically everywhere.
*/
private void atomicSetUbyte(ref ubyte stuff, ubyte newVal) {
    core.atomic.cas(cast(shared) &stuff, stuff, newVal);
}

// Cut and pasted from core.atomic.  See Bug 4760.
version(D_InlineAsm_X86) {
    private ubyte atomicReadUbyte(ref ubyte val) {
        asm {
            mov DL, 0;
            mov AL, 0;
            mov ECX, val;
            lock; // lock always needed to make this op atomic
            cmpxchg [ECX], DL;
        }
    }
} else version(D_InlineAsm_X86_64) {
    private ubyte atomicReadUbyte(ref ubyte val) {
        asm {
            mov DL, 0;
            mov AL, 0;
            mov RCX, val;
            lock; // lock always needed to make this op atomic
            cmpxchg [RCX], DL;
        }
    }
}

// This gets rid of the need for a lot of annoying casts in other parts of the
// code, when enums are involved.
private bool atomicCasUbyte(ref ubyte stuff, ubyte testVal, ubyte newVal) {
    return core.atomic.cas(cast(shared) &stuff, testVal, newVal);
}

private void atomicIncUint(ref uint num) {
    atomicOp!"+="(num, 1U);
}

//-----------------------------------------------------------------------------


/*--------------------- Generic helper functions, etc.------------------------*/
private template MapType(R, functions...) {
    static if(functions.length == 0) {
        alias typeof(unaryFun!(functions[0])(ElementType!(R).init)) MapType;
    } else {
        alias typeof(adjoin!(staticMap!(unaryFun, functions))
            (ElementType!(R).init)) MapType;
    }
}

private template ReduceType(alias fun, R, E) {
    alias typeof(binaryFun!(fun)(E.init, ElementType!(R).init)) ReduceType;
}

private template noUnsharedAliasing(T) {
    enum bool noUnsharedAliasing = !hasUnsharedAliasing!T;
}

private template isSafeTask(F) {
    enum bool isSafeTask =
    (!hasUnsharedAliasing!(ReturnType!F) ||
        (functionAttributes!F & FunctionAttribute.PURE)) &&
    !(functionAttributes!F & FunctionAttribute.REF) &&
    (isFunctionPointer!F || !hasUnsharedAliasing!F) &&
    allSatisfy!(noUnsharedAliasing, ParameterTypeTuple!F);
}

unittest {
    static assert(isSafeTask!(void function()));
    static assert(isSafeTask!(void function(uint, string)));
    static assert(!isSafeTask!(void function(uint, char[])));

    alias uint[] function(uint, string) pure F1;
    alias uint[] function(uint, string) F2;
    static assert(isSafeTask!(F1));
    static assert(!isSafeTask!(F2));
}


private void sleepMillisecond(long nMilliseconds) {
    Thread.sleep(nMilliseconds * 10_000);
}

private T* moveToHeap(T)(ref T object) {
    GC.BlkAttr gcFlags = (typeid(T).flags & 1) ?
                          cast(GC.BlkAttr) 0 :
                          GC.BlkAttr.NO_SCAN;
    T* myPtr = cast(T*) GC.malloc(T.sizeof, gcFlags);

    core.stdc.string.memcpy(myPtr, &object, T.sizeof);
    object = T.init;

    return myPtr;
}

//------------------------------------------------------------------------------
/* Various classes of task.  These use manual C-style polymorphism, the kind
 * with lots of structs and pointer casting.  This is because real classes
 * would prevent some of the allocation tricks I'm using and waste space on
 * monitors and vtbls for something that needs to be ultra-efficient.
 */

private enum TaskState : ubyte {
    notStarted,
    inProgress,
    done
}

private template BaseMixin(ubyte initTaskStatus) {
    AbstractTask* prev;
    AbstractTask* next;

    static if(is(typeof(&impl))) {
        void function(void*) runTask = &impl;
    } else {
        void function(void*) runTask;
    }

    Throwable exception;
    ubyte taskStatus = initTaskStatus;


    /* Kludge:  Some tasks need to re-submit themselves after they finish.
     * In this case, they will set themselves to TaskState.notStarted before
     * resubmitting themselves.  Setting this flag to false prevents the work
     * stealing loop from setting them to done.*/
    bool shouldSetDone = true;

    bool done() {
        if(atomicReadUbyte(taskStatus) == TaskState.done) {
            if(exception) {
                throw exception;
            }

            return true;
        }

        return false;
    }
}

// The base "class" for all of the other tasks.
private struct AbstractTask {
    mixin BaseMixin!(TaskState.notStarted);

    void job() {
        runTask(&this);
    }
}

private template AliasReturn(alias fun, T...) {
    alias AliasReturnImpl!(fun, T).ret AliasReturn;
}

private template AliasReturnImpl(alias fun, T...) {
    private T args;
    alias typeof(fun(args)) ret;
}

// Should be private, but std.algorithm.reduce is used in the zero-thread case
// and won't work w/ private.
template reduceAdjoin(functions...) {
    static if(functions.length == 1) {
        alias binaryFun!(functions[0]) reduceAdjoin;
    } else {
        T reduceAdjoin(T, U)(T lhs, U rhs) {
            alias staticMap!(binaryFun, functions) funs;

            foreach(i, Unused; typeof(lhs.field)) {
                lhs.field[i] = funs[i](lhs.field[i], rhs);
            }

            return lhs;
        }
    }
}

private template reduceFinish(functions...) {
    static if(functions.length == 1) {
        alias binaryFun!(functions[0]) reduceFinish;
    } else {


        T reduceFinish(T)(T lhs, T rhs) {
            alias staticMap!(binaryFun, functions) funs;

            foreach(i, Unused; typeof(lhs.field)) {
                lhs.field[i] = funs[i](lhs.field[i], rhs.field[i]);
            }

            return lhs;
        }
    }
}

private template ElementsCompatible(R, A) {
    static if(!isArray!A) {
        enum bool ElementsCompatible = false;
    } else {
        pragma(msg, ElementType!R.stringof ~ '\t' ~ ElementType!A.stringof);
        enum bool ElementsCompatible =
            is(ElementType!R : ElementType!A);
    }
}

/**
This struct represents the fundamental unit of work.  A $(D Task) may be
executed in parallel with any other $(D Task).  Using this struct directly
allows future/promise parallelism.  In this paradigm, a function (or delegate,
functor, etc.) is executed in a thread other than the one it was called from.
The calling thread does not block while the function is being executed.  A call
to $(D workWait()), $(D yieldWait()), or $(D spinWait()) can be used to retrive
the return value after the function is finished executing.

The proper way to create an instance of this struct is via the $(D task())
function.  See $(D task()) for usage examples.

Notes:  If a $(D Task) has been submitted to a $(D TaskPool) instance, is
being stored in a stack frame, and has not yet finished, the destructor for
this struct will automatically call $(D yieldWait()) so that the task can
finish and the stack frame can be destroyed safely.

Function results are returned from $(D yieldWait()) and friends by ref.  If
$(D fun) returns by ref, the reference will point directly to the returned
reference of $(D fun).  Otherwise it will point to a field in this struct.

Copying of this struct is disabled, since it would provide no useful semantics.
If you want to pass this struct around, you should do it by reference or
pointer.

Bugs:  Changes to $(D ref) and $(D out) arguments are not propagated to the
       call site, only to $(D args) in this struct.

       Copying is not actually disabled yet due to compiler bugs.  In the
       mean time, please understand that if you copy this struct, you're
       relying on implementation bugs.
*/
struct Task(alias fun, Args...) {
    // Work around syntactic ambiguity w.r.t. address of function return vals.
    private static T* addressOf(T)(ref T val) {
        return &val;
    }

    private static void impl(void* myTask) {
        Task* myCastedTask = cast(typeof(this)*) myTask;
        static if(is(ReturnType == void)) {
            fun(myCastedTask._args);
        } else static if(is(typeof(addressOf(fun(myCastedTask._args))))) {
            myCastedTask.returnVal = addressOf(fun(myCastedTask._args));
        } else {
            myCastedTask.returnVal = fun(myCastedTask._args);
        }
    }
    mixin BaseMixin!(TaskState.notStarted) Base;

    TaskPool pool;

    Args _args;

    /**
    The arguments the function was called with.  Changes to $(D out) and
    $(D ref) arguments will be reflected here when the function is done
    executing.
    */
    static if(__traits(isSame, fun, run)) {
        alias _args[1..$] args;
    } else {
        alias _args args;
    }

    alias typeof(fun(_args)) ReturnType;
    static if(!is(ReturnType == void)) {
        static if(is(typeof(&fun(_args)))) {
            // Ref return.
            ReturnType* returnVal;

            ref ReturnType fixRef(ReturnType* val) {
                return *val;
            }

        } else {
            ReturnType returnVal;

            ref ReturnType fixRef(ref ReturnType val) {
                return val;
            }
        }
    }

    void enforcePool() {
        enforce(this.pool !is null, "Job not submitted yet.");
    }

    private this(Args args) {
        static if(args.length > 0) {
            _args = args;
        }
    }

    /**
    If the task isn't started yet, execute it in the current thread.
    If it's done, return its return value, if any.  If it's in progress,
    busy spin until it's done, then return the return value.  If it threw
    an exception, rethrow that exception.

    This function should be used when you expect the result of the
    task to be available relatively quickly, on a timescale shorter
    than that of an OS context switch.
     */
    @property ref ReturnType spinWait() @trusted {
        enforcePool();

        this.pool.tryStealDelete( cast(AbstractTask*) &this);

        while(atomicReadUbyte(this.taskStatus) != TaskState.done) {}

        if(exception) {
            throw exception;
        }

        static if(!is(ReturnType == void)) {
            return fixRef(this.returnVal);
        }
    }

    /**
    If the task isn't started yet, execute it in the current thread.
    If it's done, return its return value, if any.  If it's in progress,
    wait on a condition variable.  If it threw an exception, rethrow that
    exception.

    This function should be used when you expect the result of the
    task to take a while, as waiting on a condition variable
    introduces latency, but results in negligible wasted CPU cycles.
     */
    @property ref ReturnType yieldWait() @trusted {
        enforcePool();
        this.pool.tryStealDelete( cast(AbstractTask*) &this);
        if(atomicReadUbyte(this.taskStatus) == TaskState.done) {

            static if(is(ReturnType == void)) {
                return;
            } else {
                return fixRef(this.returnVal);
            }
        }

        pool.lock();
        scope(exit) pool.unlock();

        while(atomicReadUbyte(this.taskStatus) != TaskState.done) {
            pool.waitUntilCompletion();
        }

        if(exception) {
            throw exception;
        }

        static if(!is(ReturnType == void)) {
            return fixRef(this.returnVal);
        }
    }

    /**
    If this $(D Task) was not started yet, execute it in the current
    thread.  If it is finished, return its result.  If it is in progress,
    execute any other available tasks from the $(D TaskPool) instance that
    this $(D Task) was submitted to until this one
    is finished.  If it threw an exception, rethrow that exception.
    If no other tasks are available, yield wait.
     */
    @property ref ReturnType workWait() @trusted {
        enforcePool();
        this.pool.tryStealDelete( cast(AbstractTask*) &this);

        while(true) {
            if(done) {
                static if(is(ReturnType == void)) {
                    return;
                } else {
                    return fixRef(this.returnVal);
                }
            }

            pool.lock();
            AbstractTask* job;
            try {
                // Locking explicitly and calling popNoSync() because
                // pop() waits on a condition variable if there are no jobs
                // in the queue.
                job = pool.popNoSync();
            } finally {
                pool.unlock();
            }

            if(job !is null) {

                version(verboseUnittest) {
                    stderr.writeln("Doing workWait work.");
                }

                pool.doJob(job);

                if(done) {
                    static if(is(ReturnType == void)) {
                        return;
                    } else {
                        return fixRef(this.returnVal);
                    }
                }
            } else {
                version(verboseUnittest) {
                    stderr.writeln("Yield from workWait.");
                }

                return yieldWait();
            }
        }
    }

    ///  Returns true if the $(D Task) is finished executing.
    @property bool done() @trusted {
        // Explicitly forwarded for documentation purposes.
        return Base.done();
    }

    @safe ~this() {
        if(pool !is null && taskStatus != TaskState.done) {
            yieldWait();
        }
    }

    // When this is uncommented, it somehow gets called even though it's
    // disabled and Bad Things Happen.
    //@disable this(this) { assert(0);}
}

/**
Creates a $(D Task) that calls an alias.

Examples:
---
// Read two files into memory at the same time.
import std.file;

// Create and submit a Task object for reading foo.txt.
auto file1Task = task!read("foo.txt");
taskPool.put(file1Task);

// Read bar.txt in parallel.
auto file2Data = read("bar.txt");

// Get the results of reading foo.txt.
auto file1Data = file1Task.yieldWait();
---

Note:
This method of creating tasks allocates on the stack and requires an explicit
submission to the $(D TaskPool) using $(D TaskPool.put()).  It is designed for
tasks that are to finish before the function in which they are created returns.
If you want to escape the Task object from the function in which it was created
or prefer to heap allocate and automatically submit to the pool, see
$(D TaskPool.task()).
 */
Task!(fun, Args) task(alias fun, Args...)(Args args) {
    alias Task!(fun, Args) RetType;
    return RetType(args);
}

/**
Calls a delegate or function pointer with $(D args).  This is an
adapter that makes $(D Task) work with delegates, function pointers and
functors instead of just aliases.  It is included in the documentation
to clarify how this case is handled, but is not meant to be used directly
by client code.
 */
ReturnType!(F) run(F, Args...)(F fpOrDelegate, ref Args args) {
    return fpOrDelegate(args);
}

/**
Create a task that calls a function pointer, delegate, or functor.

Examples:
---
// Read two files in at the same time again, but this time use a function
// pointer instead of an alias to represent std.file.read.
import std.file;

// Create and submit a Task object for reading foo.txt.
auto file1Task = task(&read, "foo.txt");
taskPool.put(file1Task);

// Read bar.txt in parallel.
auto file2Data = read("bar.txt");

// Get the results of reading foo.txt.
auto file1Data = file1Task.yieldWait();
---

Notes:
This method of creating tasks allocates on the stack and requires an explicit
submission to the $(D TaskPool) using $(D TaskPool.put()).  It is designed for
tasks that are to finish before the function in which they are created returns.
If you want to escape the Task object from the function in which it was created
or prefer to heap allocate and automatically submit to the pool, see
$(D TaskPool.task()).

In the case of delegates, this function takes a $(D scope) delegate to prevent
the allocation of closures, since its intended use is for tasks that will
be finished before the function in which they're created returns.
$(D TaskPool.task()) takes a non-scope delegate and will allow the use of
closures.
 */
Task!(run, TypeTuple!(F, Args))
task(F, Args...)(scope F delegateOrFp, Args args)
if(is(typeof(delegateOrFp(args))) && !isSafeTask!F) {
    alias typeof(return) RT;
    return RT(delegateOrFp, args);
}

/**
Safe version of Task, usable from $(D @safe) code.  Usage mechanics are
identical to the non-@safe case, but safety introduces the following
restrictions:

1.  $(D F) must not have any unshared aliasing.  Basically, this means that it
    may not be an unshared delegate.  This also precludes accepting template
    alias paramters.

2.  $(D Args) must not have unshared aliasing.

3.  The return type must not have unshared aliasing unless $(D fun) is weakly
    pure.

4.  $(D fun) must return by value, not by reference.
*/
@trusted Task!(run, TypeTuple!(F, Args))
task(F, Args...)(F fun, Args args)
if(is(typeof(fun(args))) && isSafeTask!F) {
    alias typeof(return) RT;
    return RT(fun, args);
}

/**
The total number of CPUs available on the current machine, as reported by
the operating system.
*/
immutable uint osReportedNcpu;

/**
This class encapsulates a task queue and a set of worker threads.  A task
queue is a simple FIFO queue of $(D Task) objects that have been submitted
to the $(D TaskPool) but for which execution has not yet begun.  A worker
thread is a thread that executes the $(D Task) at the front of the task queue
when one is available and sleeps when the task queue is empty.

This class should usually be used via the default global instantiation,
available via the global $(D taskPool) property, which is described below.
Occasionally it may be useful to explicitly instantiate a $(D TaskPool) object,
for example when the threads in the global task pool are blocked waiting for
some code to execute, and you want to parallelize the code that these threads
are waiting on.
 */
final class TaskPool {
private:
    Thread[] pool;
    AbstractTask* head;
    AbstractTask* tail;
    PoolState status = PoolState.running;  // All operations on this are done atomically.
    Condition workerCondition;
    Condition waiterCondition;
    Mutex mutex;

    // The instanceStartIndex of the next instance that will be created.
    __gshared static size_t nextInstanceIndex = 1;

    // The index of the current thread.
    static size_t threadIndex;

    // The index of the first thread in the next instance.
    immutable size_t instanceStartIndex;

    // The index that the next thread to be initialized in this pool will have.
    size_t nextThreadIndex;

    enum PoolState : ubyte {
        running,
        finishing,
        stopNow
    }

    void doJob(AbstractTask* job) {
        assert(job.taskStatus == TaskState.inProgress);
        assert(job.next is null);
        assert(job.prev is null);

        scope(exit) {
            lock();
            notifyWaiters();
            unlock();
        }

        try {
            job.job();
            if(job.shouldSetDone) {
                atomicSetUbyte(job.taskStatus, TaskState.done);
            }
        } catch(Throwable e) {
            job.exception = e;
            if(job.shouldSetDone) {
                atomicSetUbyte(job.taskStatus, TaskState.done);
            }
        }
    }

    void workLoop() {
        // Initialize thread index.
        lock();
        threadIndex = nextThreadIndex;
        nextThreadIndex++;
        unlock();

        while(atomicReadUbyte(status) != PoolState.stopNow) {
            AbstractTask* task = pop();
            if (task is null) {
                if(atomicReadUbyte(status) == PoolState.finishing) {
                    atomicSetUbyte(status, PoolState.stopNow);
                    return;
                }
            } else {
                doJob(task);
            }
        }
    }

    bool deleteItem(AbstractTask* item) {
        lock();
        auto ret = deleteItemNoSync(item);
        unlock();
        return ret;
    }

    bool deleteItemNoSync(AbstractTask* item)
    out {
        assert(item.next is null);
        assert(item.prev is null);
    } body {
        if(item.taskStatus != TaskState.notStarted) {
            return false;
        }
        item.taskStatus = TaskState.inProgress;

        if(item is head) {
            // Make sure head gets set properly.
            popNoSync();
            return true;;
        }
        if(item is tail) {
            tail = tail.prev;
            if(tail !is null) {
                tail.next = null;
            }
            item.next = null;
            item.prev = null;
            return true;
        }
        if(item.next !is null) {
            assert(item.next.prev is item);  // Check queue consistency.
            item.next.prev = item.prev;
        }
        if(item.prev !is null) {
            assert(item.prev.next is item);  // Check queue consistency.
            item.prev.next = item.next;
        }
        item.next = null;
        item.prev = null;
        return true;
    }

    // Pop a task off the queue.  Should only be called by worker threads.
    AbstractTask* pop() {
        lock();
        auto ret = popNoSync();
        while(ret is null && status == PoolState.running) {
            wait();
            ret = popNoSync();
        }
        unlock();
        return ret;
    }

    AbstractTask* popNoSync()
    out(returned) {
        /* If task.prev and task.next aren't null, then another thread
         * can try to delete this task from the pool after it's
         * alreadly been deleted/popped.
         */
        if(returned !is null) {
            assert(returned.next is null);
            assert(returned.prev is null);
        }
    } body {
        AbstractTask* returned = head;
        if (head !is null) {
            head = head.next;
            returned.prev = null;
            returned.next = null;
            returned.taskStatus = TaskState.inProgress;
        }
        if(head !is null) {
            head.prev = null;
        }

        return returned;
    }

    // Push a task onto the queue.
    void abstractPut(AbstractTask* task) {
        lock();
        abstractPutNoSync(task);
        unlock();
    }

    void abstractPutNoSync(AbstractTask* task)
    out {
        assert(tail.prev !is tail);
        assert(tail.next is null, text(tail.prev, '\t', tail.next));
        if(tail.prev !is null) {
            assert(tail.prev.next is tail, text(tail.prev, '\t', tail.next));
        }
    } body {
        task.next = null;
        if (head is null) { //Queue is empty.
            head = task;
            tail = task;
            tail.prev = null;
        } else {
            task.prev = tail;
            tail.next = task;
            tail = task;
        }
        notify();
    }

    // Same as trySteal, but also deletes the task from the queue so the
    // Task object can be recycled.
    bool tryStealDelete(AbstractTask* toSteal) {
        if( !deleteItem(toSteal) ) {
            return false;
        }

        toSteal.job();

        /* shouldSetDone should always be true except if the task re-submits
         * itself to the pool and needs to bypass this.*/
        if(toSteal.shouldSetDone == 1) {
            atomicSetUbyte(toSteal.taskStatus, TaskState.done);
        }

        return true;
    }

    size_t defaultBlockSize(size_t rangeLen) const pure nothrow {
        if(this.size == 0) {
            return rangeLen;
        }

        immutable size_t fourSize = 4 * (this.size + 1);
        return (rangeLen / fourSize) + ((rangeLen % fourSize == 0) ? 0 : 1);
    }

    void lock() {
        mutex.lock();
    }

    void unlock() {
        mutex.unlock();
    }

    void wait() {
        workerCondition.wait();
    }

    void notify() {
        workerCondition.notify();
    }

    void notifyAll() {
        workerCondition.notifyAll();
    }

    void waitUntilCompletion() {
        waiterCondition.wait();
    }

    void notifyWaiters() {
        waiterCondition.notifyAll();
    }

    /*
    Gets the index of the current thread relative to this pool.  Any thread
    not in this pool will receive an index of 0.  The worker threads in
    this pool receive indices of 1 through this.size().

    The worker index is useful mainly for maintaining worker-local storage.

    BUGS:  Subject to integer overflow errors if more than size_t.max worker
           threads are ever created during the course of a program's execution.
           This will likely never be fixed because it's an extreme corner case
           on 32-bit and it's completely implausible on 64-bit.
     */
    size_t workerIndex() {
        immutable rawInd = threadIndex;
        return (rawInd >= instanceStartIndex &&
                rawInd < instanceStartIndex + size) ?
                (rawInd - instanceStartIndex + 1) : 0;
    }

public:

    /**
    Default constructor that initializes a $(D TaskPool) with
    $(D osReportedNcpu) - 1.  The minus 1 is included because the main thread
    will also be available to do work.

    Note:  In the case of a single-core machine, the primitives provided
           by $(D TaskPool) will operate transparently in single-threaded mode.
     */
    this() @trusted {
        this(osReportedNcpu - 1);
    }

    /**
    Allows for custom number of worker threads.
    */
    this(size_t nWorkers) @trusted {
        synchronized(TaskPool.classinfo) {
            instanceStartIndex = nextInstanceIndex;

            // The first worker thread to be initialized will have this index,
            // and will increment it.  The second worker to be initialized will
            // have this index plus 1.
            nextThreadIndex = instanceStartIndex;
            nextInstanceIndex += nWorkers;
        }

        mutex = new Mutex(this);
        workerCondition = new Condition(mutex);
        waiterCondition = new Condition(mutex);

        pool = new Thread[nWorkers];
        foreach(ref poolThread; pool) {
            poolThread = new Thread(&workLoop);
            poolThread.start();
        }
    }

    /**
    Implements a parallel foreach loop over a range.  This works by implicitly
    creating and submitting one $(D Task) to the $(D TaskPool) for each work unit.
    A work unit may process one or more elements of $(D range).  The number of
    elements processed per work unit is controlled by the $(D workUnitSize)
    parameter.  Smaller work units provide better load balancing, but larger
    work units can avoid the overhead of creating submitting large numbers of
    $(D Task) objects.

    Examples:
    ---
    // Find the logarithm of every number from 1 to 1_000_000 in parallel.
    auto logs = new double[1_000_000];

    // Parallel foreach works with or without an index variable.  It can be
    // iterate by ref if front() of the range being iterated over
    // returns by ref.
    foreach(i, ref elem; taskPool.parallel(logs, 100)) {
        // Iterate over logs using work units of size 100.
        elem = log(i + 1.0);
    }
    ---

    Notes:

    Breaking from a parallel foreach loop breaks from the current work unit,
    but still executes other work units.  A goto from inside the parallel
    foreach loop to a label outside the loop will result in undefined
    behavior.

    In the case of non-random access ranges, parallel foreach is still usable
    but buffers lazily to an array of size $(D workUnitSize) before executing
    the parallel portion of the loop.  The exception is that, if a parallel
    foreach is executed over a range returned by $(D asyncBuf) or $(D lazyMap),
    the copying is elided and the buffers are simply swapped.  However, note
    that in this case the $(D workUnitSize) parameter of this function will be
    ignored and the work unit size will be set to the buffer size of the object
    returned by $(D asyncBuf) or $(D lazyMap).
     */
    ParallelForeach!R parallel(R)(R range, size_t workUnitSize) {
        alias ParallelForeach!R RetType;
        return RetType(this, range, workUnitSize);
    }

    /**
    Parallel foreach with default work unit size.  For ranges that don't have
    a length, the default is 512 elements.  For ranges that do, the default
    is whatever number would create exactly 4 times as many work units as
    we have worker threads.

    Examples:
    ---
    // Same example as the overload, but elide explicit specification of
    // work unit size.
    auto logs = new double[1_000_000];

    foreach(i, ref elem; taskPool.parallel(logs)) {
        elem = log(i + 1.0);
    }
    ---
     */
    ParallelForeach!R parallel(R)(R range) {
        static if(hasLength!R) {
            // Default block size is such that we would use 2x as many
            // slots as are in this thread pool.
            size_t blockSize = defaultBlockSize(range.length);
            return parallel(range, blockSize);
        } else {
            // Just use a really, really dumb guess if the user is too lazy to
            // specify.
            return parallel(range, 512);
        }
    }

    /**
    Eager parallel map.  $(D functions) are the functions to be evaluated,
    passed as template alias parameters in a style similar to std.algorithm.
    The first argument must be a random access range.

    ---
    real[] numbers = new real[1_000_000];
    foreach(i, ref num; numbers) {
        num = i;
    }

    // Find the square roots of numbers.
    real[] squareRoots = taskPool.map!sqrt(numbers);
    ---

    Immediately after the range argument, an optional work unit size argument
    may be provided.  Work units as used by $(D map) are identical to those
    defined for parallel foreach.  If no work unit size is provided, the
    default work unit size is used.

    ---
    // Same thing, but make work units explicitly of size 100.
    real[] squareRoots = taskPool.map!sqrt(numbers, 100);
    ---

    An optional buffer for returining the results may be provided as the last
    argument.  If one is not provided, one will be automatically allocated.
    If one is provided, it must be the same length as the range.

    ---
    // Same thing, but explicitly pre-allocate a buffer.
    auto squareRoots = new real[numbers.length];
    taskPool.map!sqrt(numbers, squareRoots);

    // Multiple functions, explicit buffer, and explicit work unit size.
    auto results = new Tuple!(real, real)[numbers.length];
    taskPool.map!(sqrt, log)(numbers, 100, results);
    ---
     */
    template map(functions...) {
        ///
        auto map(Args...)(Args args) {
            static if(functions.length == 1) {
                alias unaryFun!(functions[0]) fun;
            } else {
                alias adjoin!(staticMap!(unaryFun, functions)) fun;
            }

            static if(Args.length > 1 && isArray!(Args[$ - 1]) &&
                is(MapType!(Args[0], functions) : ElementType!(Args[$ - 1]))) {
                alias args[$ - 1] buf;
                alias args[0..$ - 1] args2;
                alias Args[0..$ - 1] Args2;
            } else {
                MapType!(Args[0], functions)[] buf;
                alias args args2;
                alias Args Args2;;
            }

            static if(isIntegral!(Args2[$ - 1])) {
                static assert(args2.length == 2);
                alias args2[0] range;
                auto blockSize = cast(size_t) args2[1];
            } else {
                static assert(args2.length == 1, Args);
                alias args2[0] range;
                auto blockSize = defaultBlockSize(range.length);
            }

            alias typeof(range) R;
            immutable len = range.length;

            if(buf.length == 0) {
                // Create buffer without initializing contents.
                alias MapType!(R, functions) MT;
                GC.BlkAttr gcFlags = (typeid(MT).flags & 1) ?
                                      cast(GC.BlkAttr) 0 :
                                      GC.BlkAttr.NO_SCAN;
                auto myPtr = cast(MT*) GC.malloc(len * MT.sizeof, gcFlags);
                buf = myPtr[0..len];
            }
            enforce(buf.length == len,
                text("Can't use a user supplied buffer that's the wrong size.  ",
                "(Expected  :", len, " Got:  ", buf.length));
            if(blockSize > len) {
                blockSize = len;
            }

            // Handle as a special case:
            if(size == 0) {
                size_t index = 0;
                foreach(elem; range) {
                    buf[index++] = fun(elem);
                }
                return buf;
            }

            alias MapTask!(fun, R, typeof(buf)) MTask;
            MTask[] tasks = (cast(MTask*) alloca(this.size * MTask.sizeof * 2))
                            [0..this.size * 2];
            tasks[] = MTask.init;

            size_t curPos;
            void useTask(ref MTask task) {
                task.lowerBound = curPos;
                task.upperBound = min(len, curPos + blockSize);
                task.range = range;
                task.results = buf;
                task.pool = this;
                curPos += blockSize;

                lock();
                atomicSetUbyte(task.taskStatus, TaskState.notStarted);
                abstractPutNoSync(cast(AbstractTask*) &task);
                unlock();
            }

            ubyte doneSubmitting = 0;

            Task!(run, void delegate()) submitNextBatch;

            void submitJobs() {
                // Search for slots, then sleep.
                foreach(ref task; tasks) if(task.done) {
                    useTask(task);
                    if(curPos >= len) {
                        atomicSetUbyte(doneSubmitting, 1);
                        return;
                    }
                }

                // Now that we've submitted all the worker tasks, submit
                // the next submission task.  Synchronizing on the pool
                // to prevent the stealing thread from deleting the job
                // before it's submitted.
                lock();
                atomicSetUbyte(submitNextBatch.taskStatus, TaskState.notStarted);
                abstractPutNoSync( cast(AbstractTask*) &submitNextBatch);
                unlock();
            }

            submitNextBatch = .task(&submitJobs);

            // The submitAndSteal mixin relies on the TaskPool instance
            // being called pool.
            TaskPool pool = this;

            mixin(submitAndSteal);

            return buf;
        }
    }

    /**
    A semi-lazy parallel map that can be used for pipelining.  The map functions
    are evaluated for the first $(D bufSize) elements and stored in a
    temporary buffer and made available to $(D popFront).  Meanwhile, in the
    background a second buffer of the same size is filled.  When the first
    buffer is exhausted, it is swapped with the second buffer and filled while
    the values from what was originally the second buffer are read.  Such an
    implementation allows for elements to be written to the buffer without
    the need for atomic operations or synchronization for each write, and
    enables the mapping function to be evaluated efficiently in parallel.

    Parameters:

    range:  The range to be mapped.  This must be an input range, though it
    should preferably be a random access range to avoid needing to buffer
    to temporary array before mapping.  If the $(D range) is not random access
    it will be lazily buffered to an array of size $(D bufSize) before the map
    function is evaluated.  (For an exception to this rule, see Notes.)

    bufSize:  The size of the buffer to store the evaluated elements.

    workUnitSize:  The number of elements to evaluate in a single $(D Task).
      Must be less than or equal to $(D bufSize), and in practice should be a
      fraction of $(D bufSize) such that all worker threads can be used.  If
      the default of size_t.max is used, workUnitSize will be set to the
      pool-wide default.

    Returns:  An input range representing the results of the map.  This range
              has a length iff $(D range) has a length.

    Notes:

    If a range returned by $(D lazyMap) or $(D asyncBuf) is used as an input to
    $(D lazyMap()), then as an optimization the copying from the output buffer
    of the first range to the input buffer of the second range is elided, even
    though the ranges returned by $(D lazyMap) and $(D asyncBuf) are input
    ranges.  However, this means that the $(D bufSize) parameter passed to the
    current call to $(D lazyMap()) will be ignored and the size of the buffer
    will be the buffer size of $(D range).

    Examples:
    ---
    // Pipeline reading a file, converting each line to a number, taking the
    // logarithms of the numbers, and performing the additions necessary to
    // find the sum of the logarithms.

    auto lineRange = File("numberList.txt").byLine();
    auto dupedLines = std.algorithm.map!"a.idup"(lineRange);
    auto nums = taskPool.lazyMap!(to!double)(dupedLines);
    auto logs = taskPool.lazyMap!log10(nums);

    double sum = 0;
    foreach(elem; logs) {
        sum += elem;
    }
    ---
    */
    template lazyMap(functions...) {

        ///
        auto
        lazyMap(R)(R range, size_t bufSize = 100, size_t workUnitSize = size_t.max)
        if(isInputRange!R) {
            enforce(workUnitSize == size_t.max || workUnitSize <= bufSize,
                "Work unit size must be smaller than buffer size.");
            static if(functions.length == 1) {
                alias unaryFun!(functions[0]) fun;
            } else {
                 alias adjoin!(staticMap!(unaryFun, functions)) fun;
            }

            static final class LazyMap {
                // This is a class because the task needs to be located on the heap
                // and in the non-random access case the range needs to be on the
                // heap, too.

            private:
                alias MapType!(R, functions) E;
                E[] buf1, buf2;
                R range;
                TaskPool pool;
                Task!(run, E[] delegate(E[]), E[]) nextBufTask;
                size_t workUnitSize;
                size_t bufPos;
                bool lastTaskWaited;

                static if(isRandomAccessRange!R) {
                    alias R FromType;

                    void popRange() {
                        static if(__traits(compiles, range[0..range.length])) {
                            range = range[min(buf1.length, range.length)..range.length];
                        } else static if(__traits(compiles, range[0..$])) {
                            range = range[min(buf1.length, range.length)..$];
                        } else {
                            static assert(0, "R must have slicing for LazyMap."
                                ~ "  " ~ R.stringof ~ " doesn't.");
                        }
                    }

                } else static if(is(typeof(range.buf1)) && is(typeof(range.bufPos)) &&
                  is(typeof(range.doBufSwap()))) {

                    version(unittest) {
                        pragma(msg, "LazyRange Special Case:  "
                            ~ typeof(range).stringof);
                    }

                    alias typeof(range.buf1) FromType;
                    FromType from;

                    // Just swap our input buffer with range's output buffer and get
                    // range mapping again.  No need to copy element by element.
                    FromType dumpToFrom() {
                        assert(range.buf1.length <= from.length);
                        from.length = range.buf1.length;
                        swap(range.buf1, from);
                        range._length -= (from.length - range.bufPos);
                        range.doBufSwap();

                        return from;
                    }

                } else {
                    alias ElementType!(R)[] FromType;

                    // The temporary array that data is copied to before being
                    // mapped.
                    FromType from;

                    FromType dumpToFrom() {
                        assert(from !is null);

                        size_t i;
                        for(; !range.empty && i < from.length; range.popFront()) {
                            from[i++] = range.front;
                        }

                        from = from[0..i];
                        return from;
                    }
                }

                static if(hasLength!R) {
                    size_t _length;

                    public @property size_t length() const pure nothrow @safe {
                        return _length;
                    }
                }

                this(R range, size_t bufSize, size_t workUnitSize, TaskPool pool) {
                    static if(is(typeof(range.buf1)) && is(typeof(range.bufPos)) &&
                    is(typeof(range.doBufSwap()))) {
                        bufSize = range.buf1.length;
                    }

                    buf1.length = bufSize;
                    buf2.length = bufSize;

                    static if(!isRandomAccessRange!R) {
                        from.length = bufSize;
                    }

                    this.workUnitSize = (workUnitSize == size_t.max) ?
                            pool.defaultBlockSize(bufSize) : workUnitSize;
                    this.range = range;
                    this.pool = pool;

                    static if(hasLength!R) {
                        _length = range.length;
                    }

                    fillBuf(buf1);
                    submitBuf2();
                }

                // The from parameter is a dummy and ignored in the random access
                // case.
                E[] fillBuf(E[] buf) {
                    static if(isRandomAccessRange!R) {
                        auto toMap = take(range, buf.length);
                        scope(success) popRange();
                    } else {
                        auto toMap = dumpToFrom();
                    }

                    buf = buf[0..min(buf.length, toMap.length)];
                    pool.map!(functions)(
                            toMap,
                            workUnitSize,
                            buf
                        );

                    return buf;
                }

                void submitBuf2() {
                    // Hack to reuse the task object.

                    nextBufTask = typeof(nextBufTask).init;
                    nextBufTask._args[0] = &fillBuf;
                    nextBufTask._args[1] = buf2;
                    pool.put(nextBufTask);
                }

                void doBufSwap() {
                    if(lastTaskWaited) {
                        // Then the range is empty.  Signal it here.
                        buf1 = null;
                        buf2 = null;

                        static if(!isRandomAccessRange!R) {
                            from = null;
                        }

                        return;
                    }

                    buf2 = buf1;
                    buf1 = nextBufTask.yieldWait();
                    bufPos = 0;

                    if(range.empty) {
                        lastTaskWaited = true;
                    } else {
                        submitBuf2();
                    }
                }

            public:
                MapType!(R, functions) front() @property {
                    return buf1[bufPos];
                }

                void popFront() {
                    static if(hasLength!R) {
                        _length--;
                    }

                    bufPos++;
                    if(bufPos >= buf1.length) {
                        doBufSwap();
                    }
                }

                static if(std.range.isInfinite!R) {
                    enum bool empty = false;
                } else {

                    bool empty() @property {
                        return buf1 is null;  // popFront() sets this when range is empty
                    }
                }
            }
            return new LazyMap(range, bufSize, workUnitSize, this);
        }
    }

    /**
    Given an input range that is expensive to iterate over, returns an
    object that asynchronously buffers the contents of
    $(D range) into a buffer of $(D bufSize) elements in a worker thread,
    while making prevously buffered elements from a second buffer, also of size
    $(D bufSize), available via the range interface of the returned
    object.  This is useful, for example, when performing expensive operations
    on the elements of ranges that represent data on a disk or network.

    Examples:
    ---
    auto lines = File("foo.txt").byLine();
    auto duped = map!"a.idup"(lines);  // Necessary b/c byLine() recycles buffer

    // Fetch more lines in the background while we process the lines already
    // read into memory into a matrix of doubles.
    double[][] matrix;
    auto asyncReader = taskPool.asyncBuf(duped);

    foreach(line; asyncReader) {
        auto ls = line.split("\t");
        matrix ~= to!(double[])(ls);
    }
    ---
    */
    auto asyncBuf(R)(R range, size_t bufSize = 100) {
        static final class AsyncBuf {
            // This is a class because the task and the range both need to be on
            // the heap.

            // The element type of R.
            alias ElementType!R E;  // Needs to be here b/c of forward ref bugs.

        private:
            E[] buf1, buf2;
            R range;
            TaskPool pool;
            Task!(run, E[] delegate(E[]), E[]) nextBufTask;
            size_t bufPos;
            bool lastTaskWaited;

            static if(hasLength!R) {
                size_t _length;

                /// Available if hasLength!(R).
                public @property size_t length() const pure nothrow @safe {
                    return _length;
                }
            }

            this(R range, size_t bufSize, TaskPool pool) {
                buf1.length = bufSize;
                buf2.length = bufSize;

                this.range = range;
                this.pool = pool;

                static if(hasLength!R) {
                    _length = range.length;
                }

                fillBuf(buf1);
                submitBuf2();
            }

            // The from parameter is a dummy and ignored in the random access
            // case.
            E[] fillBuf(E[] buf) {
                assert(buf !is null);

                size_t i;
                for(; !range.empty && i < buf.length; range.popFront()) {
                    buf[i++] = range.front;
                }

                buf = buf[0..i];
                return buf;
            }

            void submitBuf2() {
                // Hack to reuse the task object.

                nextBufTask = typeof(nextBufTask).init;
                nextBufTask._args[0] = &fillBuf;
                nextBufTask._args[1] = buf2;
                pool.put(nextBufTask);
            }

            void doBufSwap() {
                if(lastTaskWaited) {
                    // Then the range is empty.  Signal it here.
                    buf1 = null;
                    buf2 = null;
                    return;
                }

                buf2 = buf1;
                buf1 = nextBufTask.yieldWait();
                bufPos = 0;

                if(range.empty) {
                    lastTaskWaited = true;
                } else {
                    submitBuf2();
                }
            }

        public:

            E front() @property {
                return buf1[bufPos];
            }

            void popFront() {
                static if(hasLength!R) {
                    _length--;
                }

                bufPos++;
                if(bufPos >= buf1.length) {
                    doBufSwap();
                }
            }

            static if(std.range.isInfinite!R) {
                enum bool empty = false;
            } else {

                ///
                bool empty() @property {
                    return buf1 is null;  // popFront() sets this when range is empty
                }
            }
        }
        return new AsyncBuf(range, bufSize, this);
    }

    /**
    Parallel reduce.  The first argument must be the range to be reduced.
    It must offer random access and have a length.  An explicit work unit size
    (as described in the documentation for parallel foreach) may optionally be
    provided as the second argument.  If none is provided, the default work
    unit size is used.

    Note:  Because this operation is being carried out in parallel,
    $(D fun) must be associative.  For notational simplicity, let # be an
    infix operator representing $(D fun).  Then, (a # b) # c must equal
    a # (b # c).  Also note that floating point addition is not associative
    even though addition in exact arithmetic is.  Therefore, summing floating
    point numbers using this function may give different results than summing
    serially.  However, for many practical purposes floating point addition
    can be treated as associative.  This is done in some of the examples below.

    Examples:
    ---
    // Find the sum of squares of an array in parallel.
    auto myArr = array(iota(100_000));
    auto myMax = taskPool.reduce!"a + b * b"(myArr);

    // Find both the min and max of myArr.
    auto minMax = taskPool.reduce!(min, max)(myArr);
    assert(minMax.field[0] == reduce!min(myArr));
    assert(minMax.field[1] == reduce!max(myArr));

    // Parallel reduce can be combined with a lazy, random access, non-parallel
    // map, such as std.algorithm.map to interesting effect.  The
    // following example (thanks to Russel Winder) calculates pi by
    // quadrature using std.parallelism.map and TaskPool.reduce.
    // getTerm is naturally evaluated in parallel as needed by
    // TaskPool.reduce.

    immutable n = 1_000_000_000;
    immutable delta = 1.0 / n;

    real getTerm(int i) {
        immutable x = ( i - 0.5 ) * delta;
        return delta / ( 1.0 + x * x ) ;
    }

    immutable pi = 4.0 * taskPool.reduce!"a + b"(
        std.algorithm.map!getTerm(iota(n))
    );
    ---
     */
    template reduce(functions...) {

        ///
        auto reduce(Args...)(Args args) {
            alias reduceAdjoin!(functions) fun;
            alias reduceFinish!(functions) finishFun;

            static if(isIntegral!(Args[$ - 1])) {
                size_t blockSize = cast(size_t) args[$ - 1];
                alias args[0..$ - 1] args2;
                alias Args[0..$ - 1] Args2;
            } else {
                alias args args2;
                alias Args Args2;
            }

            static auto makeStartValue(Type)(Type e) {
                static if(functions.length == 1) {
                    return e;
                } else {
                    typeof(adjoin!(staticMap!(binaryFun, functions))(e, e))
                        startVal = void;
                    foreach (i, T; startVal.Types) {
                        auto p = (cast(void*) &startVal.field[i])
                            [0 .. startVal.field[i].sizeof];
                        emplace!T(p, e);
                    }

                    return startVal;
                }
            }

            static if(args2.length == 2) {
                static assert(isInputRange!(Args2[1]));
                alias args2[1] range;
                alias args2[0] startVal;

                static if(!is(typeof(blockSize))) {
                    size_t blockSize = defaultBlockSize(range.length);
                }
            } else {
                static assert(args2.length == 1);
                alias args2[0] range;

                static if(!is(typeof(blockSize))) {
                    size_t blockSize = defaultBlockSize(range.length);
                }


                enforce(!range.empty,
                    "Cannot reduce an empty range with first element as start value.");

                auto startVal = makeStartValue(range.front);
                range.popFront();
            }

            alias typeof(startVal) E;
            alias typeof(range) R;

            if(this.size == 0) {
                return std.algorithm.reduce!(fun)(startVal, range);
            }

            // Unlike the rest of the functions here, I can't use the Task object
            // recycling trick here because this has to work on non-commutative
            // operations.  After all the tasks are done executing, fun() has to
            // be applied on the results of these to get a final result, but
            // it can't be evaluated out of order.

            immutable len = range.length;
            if(blockSize > len) {
                blockSize = len;
            }

            immutable size_t nWorkUnits = (len / blockSize) +
                ((len % blockSize == 0) ? 0 : 1);
            assert(nWorkUnits * blockSize >= len);

            static E reduceOnRange
            (E startVal, R range, size_t lowerBound, size_t upperBound) {
                E result = startVal;
                foreach(i; lowerBound..upperBound) {
                    result = fun(result, range[i]);
                }
                return result;
            }

            alias Task!(reduceOnRange, E, R, size_t, size_t) RTask;
            RTask[] tasks;

            enum MAX_STACK = 512;
            immutable size_t nBytesNeeded = nWorkUnits * RTask.sizeof;

            if(nBytesNeeded < MAX_STACK) {
                tasks = (cast(RTask*) alloca(nBytesNeeded))[0..nWorkUnits];
                tasks[] = RTask.init;
            } else {
                tasks = new RTask[nWorkUnits];
            }

            size_t curPos = 0;
            void useTask(ref RTask task) {
                task.args[2] = curPos + 1; // lower bound.
                task.args[3] = min(len, curPos + blockSize);  // upper bound.
                task.args[1] = range;  // range
                task.args[0] = makeStartValue(range[curPos]);  // Start val.

                curPos += blockSize;
                put(task);
            }

            foreach(ref task; tasks) {
                useTask(task);
            }

            // Try to steal each of these.
            foreach(ref task; tasks) {
                tryStealDelete( cast(AbstractTask*) &task);
            }

            // Now that we've tried to steal every task, they're all either done
            // or in progress.  Wait on all of them.
            E result = startVal;
            foreach(ref task; tasks) {
                task.yieldWait();
                result = finishFun(result, task.returnVal);
            }
            return result;
        }
    }

    /**
    Struct for creating worker-local storage.  Worker-local storage is
    thread-local storage that exists only for worker threads in a given
    $(D TaskPool) plus a single thread outside the pool, is allocated on the
    heap in a way that avoids false sharing, and doesn't necessarily have global
    scope within any thread.  It can be accessed from any worker thread in the
    pool that created  it, and one thread outside this pool.  All threads
    outside the pool that created a given instance of worker-local storage
    share a single slot.

    Since the underlying data for this struct is heap-allocated, this struct
    has reference semantics when passed around.

    At a more concrete level, the main uses case for
    $(D WorkerLocalStorageStorage) are:

    1.  Performing parallel reductions with an imperative, as opposed to
    functional, programming style.  In this case, it's useful to treat
    $(D WorkerLocalStorageStorage) as local to each thread for only the parallel
    portion of an algorithm.

    2.  Recycling temporary buffers across iterations of a parallel foreach loop.

    Examples:
    ---
    // Calculate pi as in our example for parallel reduce, but use an
    // imperative instead of a functional style.
    immutable n = 1_000_000_000;
    immutable delta = 1.0L / n;

    auto sums = taskPool.workerLocalStorage(0.0L);
    foreach(i; parallel(iota(n))) {
        immutable x = ( i - 0.5L ) * delta;
        immutable toAdd = delta / ( 1.0 + x * x );
        sums.get = sums.get + toAdd;
    }

    // Add up the results from each worker thread.
    real pi = 0;
    foreach(threadResult; sums.toRange) {
        pi += 4.0L * threadResult;
    }
    ---
     */
    static struct WorkerLocalStorage(T) {
    private:
        TaskPool pool;
        size_t size;

        static immutable size_t cacheLineSize;
        size_t elemSize;
        bool* stillThreadLocal;

        shared static this() {
            size_t lineSize = 0;
            foreach(cachelevel; datacache) {
                if(cachelevel.lineSize > lineSize && cachelevel.lineSize < uint.max) {
                    lineSize = cachelevel.lineSize;
                }
            }

            cacheLineSize = lineSize;
        }

        static size_t roundToLine(size_t num) pure nothrow {
            if(num % cacheLineSize == 0) {
                return num;
            } else {
                return ((num / cacheLineSize) + 1) * cacheLineSize;
            }
        }

        void* data;

        void initialize(TaskPool pool) {
            this.pool = pool;
            size = pool.size + 1;
            stillThreadLocal = new bool;
            *stillThreadLocal = true;

            // Determines whether the GC should scan the array.
            auto blkInfo = (typeid(T).flags & 1) ?
                           cast(GC.BlkAttr) 0 :
                           GC.BlkAttr.NO_SCAN;

            immutable nElem = pool.size + 1;
            elemSize = roundToLine(T.sizeof);

            // The + 3 is to pad one full cache line worth of space on either side
            // of the data structure to make sure false sharing with completely
            // unrelated heap data is prevented, and to provide enough padding to
            // make sure that data is cache line-aligned.
            data = GC.malloc(elemSize * (nElem + 3), blkInfo) + elemSize;

            // Cache line align data ptr.
            data = cast(void*) roundToLine(cast(size_t) data);

            foreach(i; 0..nElem) {
                this.opIndex(i) = T.init;
            }
        }

        ref T opIndex(size_t index) {
            assert(index < size, text(index, '\t', uint.max));
            return *(cast(T*) (data + elemSize * index));
        }

        void opIndexAssign(T val, size_t index) {
            assert(index < size);
            *(cast(T*) (data + elemSize * index)) = val;
        }

    public:
        /**
        Get the current thread's instance.  Returns by reference.
        Note that calling $(D get()) from any thread
        outside the pool that created this instance will return the
        same reference, so an instance of worker-local storage should only be
        accessed from one thread outside the pool that created it.  If this
        rule is violated, undefined behavior will result.

        If assertions are enabled and $(D toRange()) has been called, then this
        WorkerLocalStorage instance is no longer worker-local and an assertion
        failure will result when calling this method.  This is not checked
        when assertions are disabled for performance reasons.
         */
        ref T get() @property {
            assert(*stillThreadLocal,
                   "Cannot call get() on this instance of WorkerLocalStorage because it" ~
                   " is no longer worker-local."
            );
            return opIndex(pool.workerIndex);
        }

        /**
        Assign a value to the current thread's instance.  This function has
        the same caveats as its overload.
        */
        void get(T val) @property {
            assert(*stillThreadLocal,
                   "Cannot call get() on this instance of WorkerLocalStorage because it" ~
                   " is no longer worker-local."
            );

            opIndexAssign(val, pool.workerIndex);
        }

        /**
        Returns a range view of the values for all threads, which can be used
        to further process the results of each thread after running the parallel
        part of your algorithm.  Do not use this method in the parallel portion
        of your algorithm.

        Calling this function will also set a flag indicating
        that this struct is no longer thread-local, and attempting to use the
        get() method again will result in an assertion failure if assertions
        are enabled.
         */
        WorkerLocalStorageRange!T toRange() @property {
            if(*stillThreadLocal) {
                *stillThreadLocal = false;

                // Make absolutely sure results are visible to all threads.
                synchronized {}
            }

           return WorkerLocalStorageRange!(T)(this);
        }
    }

    /**
    Range primitives for worker-local storage.  The purpose of this is to
    access results produced by each worker thread from a single thread once you
    are no longer using the worker-local storage from multiple threads.
    Do not use this struct in the parallel portion of your algorithm.

    The proper way to instantiate this object is to call
    $(D WorkerLocalStorage.toRange).  Once instantiated, this object behaves
    as a finite random-access range with assignable, lvalue elemends and
    a length equal to the number of worker threads in the task pool that
    created it plus 1.
     */
    static struct WorkerLocalStorageRange(T) {
    private:
        WorkerLocalStorage!T workerLocalStorage;

        size_t _length;
        size_t beginOffset;

        this(WorkerLocalStorage!(T) wl) {
            this.workerLocalStorage = wl;
            _length = wl.size;
        }

    public:
        ref T front() @property {
            return this[0];
        }

        ref T back() @property {
            return this[_length - 1];
        }

        void popFront() {
            if(_length > 0) {
                beginOffset++;
                _length--;
            }
        }

        void popBack() {
            if(_length > 0) {
                _length--;
            }
        }

        typeof(this) save() @property {
            return this;
        }

        ref T opIndex(size_t index) {
            assert(index < _length);
            return workerLocalStorage[index + beginOffset];
        }

        void opIndexAssign(T val, size_t index) {
            assert(index < _length);
            workerLocalStorage[index] = val;
        }

        typeof(this) opSlice(size_t lower, size_t upper) {
            assert(upper <= _length);
            auto newWl = this.workerLocalStorage;
            newWl.data += lower * newWl.elemSize;
            newWl.size = upper - lower;
            return typeof(this)(newWl);
        }

        bool empty() @property {
            return length == 0;
        }

        size_t length() @property {
            return _length;
        }
    }

    /**
    Create an instance of worker-local storage, initialized with a given
    value.  The value is $(D lazy) so that you can, for example, easily
    create one instance of a class for each worker.  For usage example,
    see the $(D WorkerLocalStorage) struct.
     */
    WorkerLocalStorage!(T) workerLocalStorage(T)(lazy T initialVal = T.init) {
        WorkerLocalStorage!(T) ret;
        ret.initialize(this);
        foreach(i; 0..size + 1) {
            ret[i] = initialVal;
        }
        synchronized {}  // Make sure updates are visible in all threads.
        return ret;
    }

    /**
    Terminates all threads in the pool without waiting for the queue to be
    empty.  Use only if you have waitied on every $(D Task) and therefore know
    there can't be more in queue, or if you speculatively executed some tasks
    and no longer need the results.

    Note:  Does not affect jobs that are already executing, only those
    in queue.
     */
    void stop() @trusted {
        lock();
        scope(exit) unlock();
        atomicSetUbyte(status, PoolState.stopNow);
        notifyAll();
    }

    /**
    Waits for all jobs to finish, then terminates all worker threads.  Blocks
    until all worker threads have terminated.

    Example:
    ---
    import std.file;

    auto pool = new TaskPool();
    auto task1 = task!read("foo.txt");
    pool.put(task1);
    auto task2 = task!read("bar.txt");
    pool.put(task2);
    auto task3 = task!read("baz.txt");
    pool.put(task3);

    // Call join() to guarantee that all tasks are done running, the worker
    // threads have terminated and that the results of all of the tasks can
    // be accessed without any synchronization primitives.
    pool.join();

    // Use spinWait() since the results are guaranteed to have been computed
    // and spinWait() is the cheapest of the wait functions.
    auto result1 = task1.spinWait();
    auto result2 = task2.spinWait();
    auto result3 = task3.spinWait();
    ---
    */
    void join() @trusted {
        finish();
        foreach(t; pool) {
            t.join();
        }
    }

    /**
    Instructs worker threads to terminate when the queue becomes empty as
    in $(D join()) but does not block.
     */
    void finish() @trusted {
        lock();
        scope(exit) unlock();
        atomicCasUbyte(status, PoolState.running, PoolState.finishing);
        notifyAll();
    }

    /// Returns the number of worker threads in the pool.
    @property size_t size() @safe const pure nothrow {
        return pool.length;
    }

    /**
    Put a $(D Task) object on the back of the task queue.

    Example:
    ---
    import std.file;

    // Create a task.
    auto t = task!read("foo.txt");

    // Add it to the queue to be executed.
    taskPool.put(t);
    ---

    Note:  While this function takes the address of variables that may
    potentially be on the stack, it is safe and marked as @trusted.  $(D Task)
    objects include a destructor that waits for the task to complete before
    destroying the stack frame that they are allocated on.  Therefore, it is
    impossible for the stack frame to be destroyed before the task is complete
    and out of the queue.
    */
    void put(alias fun, Args...)(ref Task!(fun, Args) task) @trusted {
        task.pool = this;
        abstractPut( cast(AbstractTask*) &task);
    }

    /**
    Turns the pool's threads into daemon threads so that, when the main threads
    of the program exit, the threads in the pool are automatically terminated.
    */
    void makeDaemon() @trusted {
        lock();
        scope(exit) unlock();
        foreach(thread; pool) {
            thread.isDaemon = true;
        }
    }

    /**
    Undoes a call to $(D makeDaemon).  Turns the threads in this pool into
    threads that will prevent a program from terminating even if the main
    thread has already terminated.
    */
    void makeAngel() @trusted {
        lock();
        scope(exit) unlock();
        foreach(thread; pool) {
            thread.isDaemon = false;
        }
    }

    /**
    These functions allow getting and setting the OS scheduling priority of
    the worker threads in this $(D TaskPool).  They simply forward to
    $(D core.thread.priority()), so a given priority value here means the
    same thing as an identical priority value in $(D core.thread).

    Note:  For a size zero pool, the getter arbitrarily returns
           $(D core.thread.Thread.PRIORITY_MIN) and the setter has no effect.
    */
    int priority() @property @trusted {
        return (size == 0) ? core.thread.Thread.PRIORITY_MIN :
                             pool[0].priority();
    }

    /// Ditto
    void priority(int newPriority) @property @trusted {
        if(size > 0) {
            foreach(t; pool) {
                t.priority(newPriority);
            }
        }
    }


    /**
    Convenience method that automatically creates a $(D Task) calling an alias
    on the GC heap and adds it to the back of the task queue.

    Returns:  A pointer to the $(D Task) object.

    Examples:
    ---
    // Read two files into memory at the same time.
    import std.file;

    // Create and submit a Task object for reading foo.txt.
    auto file1Task = taskPool.task!read("foo.txt");

    // Read bar.txt in parallel.
    auto file2Data = read("bar.txt");

    // Get the results of reading foo.txt.
    auto file1Data = file1Task.yieldWait();
    ---
     */
    Task!(fun, Args)* task(alias fun, Args...)(Args args) {
        auto stuff = .task!(fun)(args);
        auto ret = moveToHeap(stuff);
        put(*ret);
        return ret;
    }

    /**
    Convenience method that automatically creates a $(D Task) calling a delegate,
    function pointer, or functor on the GC heap and adds it to the back of
    the task queue.

    Example:
    ---
    // Read two files in at the same time again, but this time use a function
    // pointer instead of an alias to represent std.file.read.
    import std.file;

    // Create and submit a Task object for reading foo.txt.
    auto file1Task = taskPool.task(&read, "foo.txt");

    // Read bar.txt in parallel.
    auto file2Data = read("bar.txt");

    // Get the results of reading foo.txt.
    auto file1Data = file1Task.yieldWait();
    ---

    Returns:  A pointer to the $(D Task) object.

    Note:  This function takes a non-scope delegate, meaning it can be
    used with closures.  If you can't allocate a closure due to objects
    on the stack that have scoped destruction, see the global function
    $(D task()), which takes a scope delegate.
     */
     Task!(run, TypeTuple!(F, Args))*
     task(F, Args...)(F delegateOrFp, Args args)
     if(is(ReturnType!(F))) {
         auto stuff = .task(delegateOrFp, args);
         auto ptr = moveToHeap(stuff);
         put(*ptr);
         return ptr;
     }
}

/**
Returns a lazily initialized default instantiation of $(D TaskPool).
This function can safely be called concurrently from multiple non-worker
threads.  The worker threads in this pool are daemon threads, meaning that it
is not necessary to explicitly call $(D taskPool.join()), $(D taskPool.stop())
or $(D taskPool.finish()) before terminating the main thread.

One instance of this pool is shared across the entire program.
*/
 @property TaskPool taskPool() @trusted {
    static bool initialized;
    __gshared static TaskPool pool;

    if(!initialized) {
        synchronized {
            if(!pool) {
                pool = new TaskPool(defaultPoolThreads);
                pool.makeDaemon();
            }
        }

        initialized = true;
    }

    return pool;
}

private shared uint _defaultPoolThreads;
shared static this() {
    cas(&_defaultPoolThreads, _defaultPoolThreads, osReportedNcpu - 1U);
}

/**
These properties get and set the number of worker threads in the default pool
returned by $(D taskPool()).  If the setter is never called, the default value
is $(D osReportedNcpu) - 1  Any changes made via the setter after the default
pool is initialized via the first call to $(D taskPool()) have no effect
on the number of worker threads in the default pool.
*/
@property uint defaultPoolThreads() @trusted {
    // Kludge around lack of atomic load.
    return atomicOp!"+"(_defaultPoolThreads, 0U);
}

/// Ditto
@property void defaultPoolThreads(uint newVal) @trusted {
    cas(&_defaultPoolThreads, _defaultPoolThreads, newVal);
}

/**
Convenience functions that forwards to taskPool.parallel().  The only
purpose of these is to make parallel foreach less verbose and more
readable.

Example:
---
// Find the logarithm of every number from 1 to 1_000_000 in parallel,
// using the default TaskPool instance.
auto logs = new double[1_000_000];

foreach(i, ref elem; parallel(logs)) {
    elem = log(i + 1.0);
}
---

*/
ParallelForeach!R parallel(R)(R range) {
    return taskPool.parallel(range);
}

/// Ditto
ParallelForeach!R parallel(R)(R range, size_t workUnitSize) {
    return taskPool.parallel(range, workUnitSize);
}

private struct ParallelForeachTask(R, Delegate)
if(isRandomAccessRange!R && hasLength!R) {
    enum withIndex = ParameterTypeTuple!(Delegate).length == 2;

    static void impl(void* myTask) {
        auto myCastedTask = cast(ParallelForeachTask!(R, Delegate)*) myTask;
        foreach(i; myCastedTask.lowerBound..myCastedTask.upperBound) {

            static if(hasLvalueElements!R) {
                static if(withIndex) {
                    if(myCastedTask.runMe(i, myCastedTask.myRange[i])) break;
                } else {
                    if(myCastedTask.runMe( myCastedTask.myRange[i])) break;
                }
            } else {
                auto valToPass = myCastedTask.myRange[i];
                static if(withIndex) {
                    if(myCastedTask.runMe(i, valToPass)) break;
                } else {
                    if(myCastedTask.runMe(valToPass)) break;
                }
            }
        }

        // Allow some memory reclamation.
        myCastedTask.myRange = R.init;
        myCastedTask.runMe = null;
    }

    mixin BaseMixin!(TaskState.done);

    TaskPool pool;

    // More specific stuff.
    size_t lowerBound;
    size_t upperBound;
    R myRange;
    Delegate runMe;

    void wait() {
        if(pool is null) {
            // Never submitted.  No need to wait.
            return;
        }

        pool.lock();
        scope(exit) pool.unlock();

        // No work stealing here b/c the function that waits on this task
        // wants to recycle it as soon as it finishes.
        while(!done()) {
            pool.waitUntilCompletion();
        }

        if(exception) {
            throw exception;
        }
    }
}

private struct ParallelForeachTask(R, Delegate)
if(!isRandomAccessRange!R || !hasLength!R) {
    enum withIndex = ParameterTypeTuple!(Delegate).length == 2;

    static void impl(void* myTask) {
        auto myCastedTask = cast(ParallelForeachTask!(R, Delegate)*) myTask;

        static ref ElementType!(R) getElement(T)(ref T elemOrPtr) {
            static if(is(typeof(*elemOrPtr) == ElementType!R)) {
                return *elemOrPtr;
            } else {
                return elemOrPtr;
            }
        }

        foreach(i, element; myCastedTask.elements) {
            static if(withIndex) {
                size_t lValueIndex = i + myCastedTask.startIndex;
                if(myCastedTask.runMe(lValueIndex, getElement(element))) break;
            } else {
                if(myCastedTask.runMe(getElement(element))) break;
            }
        }

        // Make memory easier to reclaim.
        myCastedTask.runMe = null;
    }

    mixin BaseMixin!(TaskState.done);

    TaskPool pool;

    // More specific stuff.
    alias ElementType!R E;
    Delegate runMe;

    static if(hasLvalueElements!(R)) {
        E*[] elements;
    } else {
        E[] elements;
    }
    size_t startIndex;

    void wait() {
        if(pool is null) {
            // Never submitted.  No need to wait.
            return;
        }

        pool.lock();
        scope(exit) pool.unlock();

        // No work stealing here b/c the function that waits on this task
        // wants to recycle it as soon as it finishes.

        while(!done()) {
            pool.waitUntilCompletion();
        }

        if(exception) {
            throw exception;
        }
    }
}

private struct MapTask(alias fun, R, ReturnType)
if(isRandomAccessRange!R && hasLength!R) {
    static void impl(void* myTask) {
        auto myCastedTask = cast(MapTask!(fun, R, ReturnType)*) myTask;

        foreach(i; myCastedTask.lowerBound..myCastedTask.upperBound) {
            myCastedTask.results[i] = uFun(myCastedTask.range[i]);
        }

        // Nullify stuff, make GC's life easier.
        myCastedTask.results = null;
        myCastedTask.range = R.init;
    }

    mixin BaseMixin!(TaskState.done);

    TaskPool pool;

    // More specific stuff.
    alias unaryFun!fun uFun;
    R range;
    alias ElementType!R E;
    ReturnType results;
    size_t lowerBound;
    size_t upperBound;

    void wait() {
        if(pool is null) {
            // Never submitted.  No need to wait on it.
            return;
        }

        pool.lock();
        scope(exit) pool.unlock();

        // Again, no work stealing.

        while(!done()) {
            pool.waitUntilCompletion();
        }

        if(exception) {
            throw exception;
        }
    }
}

// Where the magic happens.  This mixin causes tasks to be submitted lazily to
// the task pool.  Attempts are then made by the calling thread to steal
// them.
private enum submitAndSteal = q{

    // See documentation for BaseMixin.shouldSetDone.
    submitNextBatch.shouldSetDone = false;

    // Submit first batch from this thread.
    submitJobs();

    while( !atomicReadUbyte(doneSubmitting) ) {
        // Try to steal parallel foreach tasks.
        foreach(ref task; tasks) {
            pool.tryStealDelete( cast(AbstractTask*) &task);
        }

        // All tasks in progress or done unless next
        // submission task started running.  Try to steal the submission task.
        pool.tryStealDelete(cast(AbstractTask*) &submitNextBatch);
    }

    // Steal one last time, after they're all submitted.
    foreach(ref task; tasks) {
        pool.tryStealDelete( cast(AbstractTask*) &task);
    }


    foreach(ref task; tasks) {
        task.wait();
    }
};

/*------Structs that implement opApply for parallel foreach.------------------*/
private template randLen(R) {
    enum randLen = isRandomAccessRange!R && hasLength!R;
}

private enum string parallelApplyMixin = q{
    alias ParallelForeachTask!(R, typeof(dg)) PTask;

    // Handle empty thread pool as special case.
    if(pool.size == 0) {
        int res = 0;
        size_t index = 0;

        // The explicit ElementType!R in the foreach loops is necessary for
        // correct behavior when iterating over strings.
        static if(hasLvalueElements!(R)) {
            foreach(ref ElementType!R elem; range) {
                static if(ParameterTypeTuple!(dg).length == 2) {
                    res = dg(index, elem);
                } else {
                    res = dg(elem);
                }
                index++;
            }
        } else {
            foreach(ElementType!R elem; range) {
                static if(ParameterTypeTuple!(dg).length == 2) {
                    res = dg(index, elem);
                } else {
                    res = dg(elem);
                }
                index++;
            }
        }
        return res;
    }

    PTask[] tasks = (cast(PTask*) alloca(pool.size * PTask.sizeof * 2))
                    [0..pool.size * 2];
    tasks[] = PTask.init;
    Task!(run, void delegate()) submitNextBatch;

    static if(is(typeof(range.buf1)) && is(typeof(range.bufPos)) &&
    is(typeof(range.doBufSwap()))) {
        enum bool bufferTrick = true;

        version(unittest) {
            pragma(msg, "Parallel Foreach Buffer Trick:  " ~ R.stringof);
        }

    } else {
        enum bool bufferTrick = false;
    }


    static if(randLen!R) {

        immutable size_t len = range.length;
        size_t curPos = 0;

        void useTask(ref PTask task) {
            task.lowerBound = curPos;
            task.upperBound = min(len, curPos + blockSize);
            task.myRange = range;
            task.runMe = dg;
            task.pool = pool;
            curPos += blockSize;

            pool.lock();
            atomicSetUbyte(task.taskStatus, TaskState.notStarted);
            pool.abstractPutNoSync(cast(AbstractTask*) &task);
            pool.unlock();
        }

        void submitJobs() {
            // Search for slots to recycle.
            foreach(ref task; tasks) if(task.done) {
                useTask(task);
                if(curPos >= len) {
                    atomicSetUbyte(doneSubmitting, 1);
                    return;
                }
            }

            // Now that we've submitted all the worker tasks, submit
            // the next submission task.  Synchronizing on the pool
            // to prevent the stealing thread from deleting the job
            // before it's submitted.
            pool.lock();
            atomicSetUbyte(submitNextBatch.taskStatus, TaskState.notStarted);
            pool.abstractPutNoSync( cast(AbstractTask*) &submitNextBatch);
            pool.unlock();
        }

    } else {

        static if(bufferTrick) {
            blockSize = range.buf1.length;
        }

        void useTask(ref PTask task) {
            task.runMe = dg;
            task.pool = pool;

            static if(bufferTrick) {
                // Elide copying by just swapping buffers.
                task.elements.length = range.buf1.length;
                swap(range.buf1, task.elements);

                static if(is(typeof(range._length))) {
                    range._length -= (task.elements.length - range.bufPos);
                }

                range.doBufSwap();

            } else {
                size_t copyIndex = 0;

                if(task.elements.length == 0) {
                    task.elements.length = blockSize;
                }

                for(; copyIndex < blockSize && !range.empty; copyIndex++) {
                    static if(hasLvalueElements!R) {
                        task.elements[copyIndex] = &range.front();
                    } else {
                        task.elements[copyIndex] = range.front;
                    }
                    range.popFront;
                }

                // We only actually change the array  size on the last task,
                // when the range is empty.
                task.elements = task.elements[0..copyIndex];
            }

            pool.lock();
            task.startIndex = this.startIndex;
            this.startIndex += task.elements.length;
            atomicSetUbyte(task.taskStatus, TaskState.notStarted);
            pool.abstractPutNoSync(cast(AbstractTask*) &task);
            pool.unlock();
        }


        void submitJobs() {
            // Search for slots to recycle.
            foreach(ref task; tasks) if(task.done) {
                useTask(task);
                if(range.empty) {
                    atomicSetUbyte(doneSubmitting, 1);
                    return;
                }
            }

            // Now that we've submitted all the worker tasks, submit
            // the next submission task.  Synchronizing on the pool
            // to prevent the stealing thread from deleting the job
            // before it's submitted.
            pool.lock();
            atomicSetUbyte(submitNextBatch.taskStatus, TaskState.notStarted);
            pool.abstractPutNoSync( cast(AbstractTask*) &submitNextBatch);
            pool.unlock();
        }

    }
    submitNextBatch = task(&submitJobs);

    mixin(submitAndSteal);

    return 0;
};

private struct ParallelForeach(R) {
    TaskPool pool;
    R range;
    size_t blockSize;
    size_t startIndex;
    ubyte doneSubmitting;

    alias ElementType!R E;

    int opApply(scope int delegate(ref E) dg) {
        mixin(parallelApplyMixin);
    }

    int opApply(scope int delegate(ref size_t, ref E) dg) {
        mixin(parallelApplyMixin);
    }
}

version(unittest) {
    // This was the only way I could get nested maps to work.
    __gshared TaskPool poolInstance;
}

// These test basic functionality but don't stress test for threading bugs.
// These are the tests that should be run every time Phobos is compiled.
unittest {
    poolInstance = new TaskPool(2);

    auto oldPriority = poolInstance.priority;
    poolInstance.priority = Thread.PRIORITY_MAX;
    assert(poolInstance.priority == Thread.PRIORITY_MAX);

    poolInstance.priority = Thread.PRIORITY_MIN;
    assert(poolInstance.priority == Thread.PRIORITY_MIN);

    poolInstance.priority = oldPriority;
    assert(poolInstance.priority == oldPriority);

    scope(exit) poolInstance.stop;

    static void refFun(ref uint num) {
        num++;
    }

    uint x;
    auto t = task!refFun(x);
    poolInstance.put(t);
    t.yieldWait();
    assert(t.args[0] == 1);

    auto t2 = task(&refFun, x);
    poolInstance.put(t2);
    t2.yieldWait();
    assert(t2.args[0] == 1);

    // Test ref return.
    uint toInc = 0;
    static ref T makeRef(T)(ref T num) {
        return num;
    }

    auto t3 = task!makeRef(toInc);
    taskPool.put(t3);//.submit;
    assert(t3.args[0] == 0);
    t3.spinWait++;
    assert(t3.args[0] == 1);

    static void testSafe() @safe {
        static int bump(int num) {
            return num + 1;
        }

        auto safePool = new TaskPool(0);
        auto t = task(&bump, 1);
        taskPool.put(t);
        assert(t.yieldWait == 2);
        safePool.stop;
    }

    auto arr = [1,2,3,4,5];
    auto appNums = appender!(uint[])();
    auto appNums2 = appender!(uint[])();
    foreach(i, ref elem; poolInstance.parallel(arr)) {
        elem++;
        synchronized {
            appNums.put(cast(uint) i + 2);
            appNums2.put(elem);
        }
    }

    uint[] nums = appNums.data, nums2 = appNums2.data;
    sort!"a.at!0 < b.at!0"(zip(nums, nums2));
    assert(nums == [2,3,4,5,6]);
    assert(nums2 == nums);
    assert(arr == nums);

    // Test parallel foreach with non-random access range.
    nums = null;
    nums2 = null;
    auto range = filter!"a != 666"([0, 1, 2, 3, 4]);
    foreach(i, elem; poolInstance.parallel(range)) {
        synchronized {
            nums ~= cast(uint) i;
            nums2 ~= cast(uint) i;
        }
    }

    sort!"a.at!0 < b.at!0"(zip(nums, nums2));
    assert(nums == nums2);
    assert(nums == [0,1,2,3,4]);


    assert(poolInstance.map!"a * a"([1,2,3,4,5]) == [1,4,9,16,25]);
    assert(poolInstance.map!("a * a", "-a")([1,2,3]) ==
        [tuple(1, -1), tuple(4, -2), tuple(9, -3)]);

    auto tupleBuf = new Tuple!(int, int)[3];
    poolInstance.map!("a * a", "-a")([1,2,3], tupleBuf);
    assert(tupleBuf == [tuple(1, -1), tuple(4, -2), tuple(9, -3)]);
    poolInstance.map!("a * a", "-a")([1,2,3], 5, tupleBuf);
    assert(tupleBuf == [tuple(1, -1), tuple(4, -2), tuple(9, -3)]);

    auto buf = new int[5];
    poolInstance.map!"a * a"([1,2,3,4,5], buf);
    assert(buf == [1,4,9,16,25]);
    poolInstance.map!"a * a"([1,2,3,4,5], 4, buf);
    assert(buf == [1,4,9,16,25]);


    assert(poolInstance.reduce!"a + b"([1,2,3,4]) == 10);
    assert(poolInstance.reduce!"a + b"(5.0, [1,2,3,4]) == 15);
    assert(poolInstance.reduce!(min, max)([1,2,3,4]) == tuple(1, 4));
    assert(poolInstance.reduce!("a + b", "a * b")(tuple(5, 2), [1,2,3,4]) ==
        tuple(15, 48));

    // Test worker-local storage.
    auto wl = poolInstance.workerLocalStorage(0);
    foreach(i; poolInstance.parallel(iota(1000), 1)) {
        wl.get = wl.get + i;
    }

    auto wlRange = wl.toRange;
    auto parallelSum = poolInstance.reduce!"a + b"(wlRange);
    assert(parallelSum == 499500);
    assert(wlRange[0..1][0] == wlRange[0]);
    assert(wlRange[1..2][0] == wlRange[1]);

    // Test default pool stuff.
    assert(taskPool.size == osReportedNcpu - 1);

    nums = null;
    foreach(i; parallel(iota(1000))) {
        synchronized {
            nums ~= i;
        }
    }
    sort(nums);
    assert(equal(nums, iota(1000)));

    assert(equal(
        poolInstance.lazyMap!"a * a"(iota(30_000_001), 10_000, 1000),
        std.algorithm.map!"a * a"(iota(30_000_001))
    ));

    // The filter is to kill random access and test the non-random access
    // branch.
    assert(equal(
        poolInstance.lazyMap!"a * a"(
            filter!"a == a"(iota(30_000_001)
        ), 10_000, 1000),
        std.algorithm.map!"a * a"(iota(30_000_001))
    ));

    assert(
        reduce!"a + b"(0UL,
            poolInstance.lazyMap!"a * a"(iota(3_000_001), 10_000)
        ) ==
        reduce!"a + b"(0UL,
            std.algorithm.map!"a * a"(iota(3_000_001))
        )
    );

    assert(equal(
        iota(1_000_002),
        poolInstance.asyncBuf(filter!"a == a"(iota(1_000_002)))
    ));

    // Test LazyMap/AsyncBuf chaining.
    auto lmchain = poolInstance.lazyMap!"a * a"(
        poolInstance.lazyMap!sqrt(
            poolInstance.asyncBuf(
                iota(3_000_000)
            )
        )
    );

    foreach(i, elem; parallel(lmchain)) {
        assert(approxEqual(elem, i));
    }

    auto myTask = task!(std.math.abs)(-1);
    taskPool.put(myTask);
    assert(myTask.spinWait == 1);

    // Test that worker local storage from one pool receives an index of 0
    // when the index is queried w.r.t. another pool.  The only way to do this
    // is non-deterministically.
    foreach(i; parallel(iota(1000), 1)) {
        assert(poolInstance.workerIndex == 0);
    }

    foreach(i; poolInstance.parallel(iota(1000), 1)) {
        assert(taskPool.workerIndex == 0);
    }
}

//version = parallelismStressTest;

// These are more like stress tests than real unit tests.  They print out
// tons of stuff and should not be run every time make unittest is run.
version(parallelismStressTest) {
    // These unittests are intended to also function as an example of how to
    // use this module.
    unittest {
        size_t attempt;
        for(; attempt < 10; attempt++)
        foreach(poolSize; [0, 4]) {

            // Create a TaskPool object with the default number of threads.
            poolInstance = new TaskPool(poolSize);

            // Create some data to work on.
            uint[] numbers = new uint[1_000];

            // Fill in this array in parallel, using default block size.
            // Note:  Be careful when writing to adjacent elements of an arary from
            // different threads, as this can cause word tearing bugs when
            // the elements aren't properly aligned or aren't the machine's native
            // word size.  In this case, though, we're ok.
            foreach(i; poolInstance.parallel( iota(0, numbers.length)) ) {
                numbers[i] = cast(uint) i;
            }

            // Make sure it works.
            foreach(i; 0..numbers.length) {
                assert(numbers[i] == i);
            }

            stderr.writeln("Done creating nums.");

            // Parallel foreach also works on non-random access ranges, albeit
            // less efficiently.
            auto myNumbers = filter!"a % 7 > 0"( iota(0, 1000));
            foreach(num; poolInstance.parallel(myNumbers)) {
                assert(num % 7 > 0 && num < 1000);
            }
            stderr.writeln("Done modulus test.");

            // Use parallel map to calculate the square of each element in numbers,
            // and make sure it's right.
            uint[] squares = poolInstance.map!"a * a"(numbers, 100);
            assert(squares.length == numbers.length);
            foreach(i, number; numbers) {
                assert(squares[i] == number * number);
            }
            stderr.writeln("Done squares.");

            // Sum up the array in parallel with the current thread.
            auto sumFuture = poolInstance.task!( reduce!"a + b" )(numbers);

            // Go off and do other stuff while that future executes:
            // Find the sum of squares of numbers.
            ulong sumSquares = 0;
            foreach(elem; numbers) {
                sumSquares += elem * elem;
            }

            // Ask for our result.  If the pool has not yet started working on
            // this task, spinWait() automatically steals it and executes it in this
            // thread.
            uint mySum = sumFuture.spinWait();
            assert(mySum == 999 * 1000 / 2);

            // We could have also computed this sum in parallel using parallel
            // reduce.
            auto mySumParallel = poolInstance.reduce!"a + b"(numbers);
            assert(mySum == mySumParallel);
            stderr.writeln("Done sums.");

            // Execute an anonymous delegate as a task.
            auto myTask = task({
                synchronized writeln("Our lives are parallel...Our lives are parallel.");
            });
            poolInstance.put(myTask);

            // Parallel foreach loops can also be nested, and can have an index
            // variable attached to the foreach loop.
            auto nestedOuter = "abcd";
            auto nestedInner =  iota(0, 10, 2);

            foreach(i, letter; poolInstance.parallel(nestedOuter, 1)) {
                foreach(j, number; poolInstance.parallel(nestedInner, 1)) {
                    synchronized writeln
                        (i, ": ", letter, "  ", j, ": ", number);
                }
            }

            // Block until all jobs are finished and then shut down the thread pool.
            poolInstance.join();
        }

        assert(attempt == 10);
        writeln("Press enter to go to next round of unittests.");
        readln();
    }

    // These unittests are intended more for actual testing and not so much
    // as examples.
    unittest {
        foreach(attempt; 0..10)
        foreach(poolSize; [0, 4]) {
            poolInstance = new TaskPool(poolSize);

            // Test indexing.
            stderr.writeln("Creator Raw Index:  ", poolInstance.threadIndex);
            assert(poolInstance.workerIndex() == 0);

            // Test worker-local storage.
            auto workerLocalStorage = poolInstance.workerLocalStorage!(uint)(1);
            foreach(i; poolInstance.parallel(iota(0U, 1_000_000))) {
                workerLocalStorage.get++;
            }
            assert(reduce!"a + b"(workerLocalStorage.toRange) ==
                1_000_000 + poolInstance.size + 1);

            // Make sure work is reasonably balanced among threads.  This test is
            // non-deterministic and is more of a sanity check than something that
            // has an absolute pass/fail.
            uint[void*] nJobsByThread;
            foreach(thread; poolInstance.pool) {
                nJobsByThread[cast(void*) thread] = 0;
            }
            nJobsByThread[ cast(void*) Thread.getThis] = 0;

            foreach(i; poolInstance.parallel( iota(0, 1_000_000), 100 )) {
                atomicIncUint( nJobsByThread[ cast(void*) Thread.getThis() ]);
            }

            stderr.writeln("\nCurrent (stealing) thread is:  ",
                cast(void*) Thread.getThis());
            stderr.writeln("Workload distribution:  ");
            foreach(k, v; nJobsByThread) {
                stderr.writeln(k, '\t', v);
            }

            // Test whether map can be nested.
            real[][] matrix = new real[][](1000, 1000);
            foreach(i; poolInstance.parallel( iota(0, matrix.length) )) {
                foreach(j; poolInstance.parallel( iota(0, matrix[0].length) )) {
                    matrix[i][j] = i * j;
                }
            }

            // Get around weird bugs having to do w/ sqrt being an intrinsic:
            static real mySqrt(real num) {
                return sqrt(num);
            }

            static real[] parallelSqrt(real[] nums) {
                return poolInstance.map!mySqrt(nums);
            }

            real[][] sqrtMatrix = poolInstance.map!parallelSqrt(matrix);

            foreach(i, row; sqrtMatrix) {
                foreach(j, elem; row) {
                    real shouldBe = sqrt( cast(real) i * j);
                    assert(approxEqual(shouldBe, elem));
                    sqrtMatrix[i][j] = shouldBe;
                }
            }

            auto saySuccess = task({
                stderr.writeln(
                    "Success doing matrix stuff that involves nested pool use.");
            });
            poolInstance.put(saySuccess);
            saySuccess.workWait();

            // A more thorough test of map, reduce:  Find the sum of the square roots of
            // matrix.

            static real parallelSum(real[] input) {
                return poolInstance.reduce!"a + b"(input);
            }

            auto sumSqrt = poolInstance.reduce!"a + b"(
                poolInstance.map!parallelSum(
                    sqrtMatrix
                )
            );

            assert(approxEqual(sumSqrt, 4.437e8));
            stderr.writeln("Done sum of square roots.");

            // Test whether tasks work with function pointers.
            auto nanTask = poolInstance.task(&isNaN, 1.0L);
            assert(nanTask.spinWait == false);

            if(poolInstance.size > 0) {
                // Test work waiting.
                static void uselessFun() {
                    foreach(i; 0..1_000_000) {}
                }

                auto uselessTasks = new typeof(task(&uselessFun))[1000];
                foreach(ref uselessTask; uselessTasks) {
                    uselessTask = task(&uselessFun);
                }
                foreach(ref uselessTask; uselessTasks) {
                    poolInstance.put(uselessTask);
                }
                foreach(ref uselessTask; uselessTasks) {
                    uselessTask.workWait();
                }
            }

            // Test the case of non-random access + ref returns.
            int[] nums = [1,2,3,4,5];
            static struct RemoveRandom {
                int[] arr;

                ref int front() { return arr.front; }
                void popFront() { arr.popFront(); }
                bool empty() { return arr.empty; }
            }

            auto refRange = RemoveRandom(nums);
            foreach(ref elem; poolInstance.parallel(refRange)) {
                elem++;
            }
            assert(nums == [2,3,4,5,6]);
            stderr.writeln("Nums:  ", nums);

            poolInstance.stop();
        }
    }
}
void main(){}
