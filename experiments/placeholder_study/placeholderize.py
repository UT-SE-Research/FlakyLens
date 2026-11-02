"""Step 2 - replace project-specific identifiers with canonical placeholders.

Goal: remove *which codebase this is* while preserving *what the test does*, so we can
test whether FlakyLens is classifying flakiness or project identity.

Method
------
javalang.tokenizer gives every token a (line, column) position, and survives snippets
that javalang.parse rejects (~13% of FlakeBench methods are truncated mid-body). We
tokenize, classify each Identifier, and rewrite right-to-left so earlier positions stay
valid. Where the full parse succeeds we use the AST to know which names are declared
methods vs locals; otherwise we fall back to a positional heuristic.

Classification of an Identifier token:
  in ALLOW                          -> keep verbatim
  inside a package/import statement -> PKG_n
  declared method, or followed by ( -> METHOD_n
  starts uppercase                  -> CLASS_n
  otherwise                         -> VAR_n

Mapping is consistent within a test (same name -> same placeholder everywhere) and
independent across tests, so no cross-test identity survives either.

NOT touched: keywords, literals, operators, annotations' @ symbol, and every name in
ALLOW. String literals are deliberately left alone - see CAVEAT at the bottom.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict

import javalang

csv.field_size_limit(10 ** 9)

# --------------------------------------------------------------------------------------
# Allowlist: standard + well-known third-party vocabulary that is NOT project-specific.
# Anything here is preserved verbatim, so concurrency/wait semantics survive intact.
# --------------------------------------------------------------------------------------
JAVA_LANG = """String Integer Long Boolean Double Float Object Exception RuntimeException
Throwable Thread Runnable System Math Character Byte Short Void Class Comparable Iterable
StringBuilder StringBuffer Number Error InterruptedException IllegalStateException
IllegalArgumentException NullPointerException UnsupportedOperationException AutoCloseable
ClassNotFoundException Cloneable Process ProcessBuilder ThreadLocal ThreadGroup
IndexOutOfBoundsException ArrayIndexOutOfBoundsException NumberFormatException""".split()

JAVA_UTIL = """List ArrayList Map HashMap Set HashSet Collection Collections Arrays Iterator
Optional Objects LinkedList TreeMap TreeSet LinkedHashMap LinkedHashSet Queue Deque
ArrayDeque Random UUID Date Calendar Comparator Stream Collectors IntStream Function
Supplier Consumer Predicate BiFunction BiConsumer Entry Properties Scanner Locale
NoSuchElementException ConcurrentModificationException""".split()

# concurrency: the whole point of the study - must survive
CONCURRENT = """ExecutorService Executors Executor Future CompletableFuture CountDownLatch
CyclicBarrier Semaphore TimeUnit ConcurrentHashMap ConcurrentMap ConcurrentLinkedQueue
ConcurrentLinkedDeque BlockingQueue LinkedBlockingQueue LinkedBlockingDeque ArrayBlockingQueue
SynchronousQueue ThreadPoolExecutor ScheduledExecutorService ScheduledFuture TimeoutException
ExecutionException CancellationException RejectedExecutionException Callable ThreadLocalRandom
ForkJoinPool ForkJoinTask RecursiveTask Phaser Exchanger CopyOnWriteArrayList CopyOnWriteArraySet
ConcurrentSkipListMap CountedCompleter Flow AtomicInteger AtomicLong AtomicBoolean AtomicReference
AtomicIntegerArray AtomicLongArray AtomicReferenceArray LongAdder DoubleAdder LongAccumulator
Lock ReentrantLock ReadWriteLock ReentrantReadWriteLock Condition LockSupport StampedLock""".split()

# concurrency / waiting / timing verbs and constants
VERBS = """sleep wait notify notifyAll await join start run interrupt interrupted isInterrupted
isAlive setDaemon isDaemon setName getName currentThread yield onSpinWait countDown getCount
acquire release tryAcquire lock unlock tryLock lockInterruptibly newCondition signal signalAll
submit execute invokeAll invokeAny shutdown shutdownNow awaitTermination isShutdown isTerminated
schedule scheduleAtFixedRate scheduleWithFixedDelay compareAndSet weakCompareAndSet getAndIncrement
incrementAndGet getAndDecrement decrementAndGet getAndAdd addAndGet getAndSet accumulateAndGet
updateAndGet currentTimeMillis nanoTime complete completeExceptionally isDone isCancelled cancel
thenApply thenAccept thenRun thenCompose allOf anyOf supplyAsync runAsync
MILLISECONDS SECONDS NANOSECONDS MICROSECONDS MINUTES HOURS DAYS
Duration Instant LocalDateTime LocalDate LocalTime Clock Timer TimerTask""".split()

JUNIT = """Test Before After BeforeClass AfterClass BeforeEach AfterEach Assert Assertions
assertEquals assertTrue assertFalse assertNull assertNotNull assertThat assertArrayEquals
assertSame assertNotSame assertThrows assertDoesNotThrow assertAll assertTimeout fail
Ignore Disabled DisplayName Rule ClassRule RunWith Parameterized Parameters Timeout TestRule
ExpectedException TemporaryFolder ParameterizedTest ValueSource MethodSource CsvSource
RepeatedTest TestInfo TestInstance Nested Tag Order TestMethodOrder""".split()

THIRD_PARTY = """Mock Mockito InjectMocks Spy Captor ArgumentCaptor MockitoAnnotations
mock spy when verify times never atLeast atLeastOnce atMost doReturn doThrow doAnswer doNothing
thenReturn thenThrow thenAnswer any anyString anyInt anyLong anyBoolean eq same isA argThat
Matchers MatcherAssert CoreMatchers is equalTo not nullValue notNullValue instanceOf hasSize
hasItem hasItems contains containsInAnyOrder greaterThan lessThan greaterThanOrEqualTo
Awaitility await atMost until untilAsserted untilTrue pollInterval pollDelay
Logger LoggerFactory Slf4j Log LogFactory""".split()

MODIFIERS = """Override SuppressWarnings Deprecated FunctionalInterface SafeVarargs Nullable
NonNull Nonnull VisibleForTesting Beta Internal Experimental value""".split()

ALLOW = set(JAVA_LANG + JAVA_UTIL + CONCURRENT + VERBS + JUNIT + THIRD_PARTY + MODIFIERS)

# common package roots to keep so `java.util.concurrent.X` stays readable as JDK
PKG_ALLOW = {"java", "javax", "util", "concurrent", "atomic", "locks", "lang", "io", "nio",
             "time", "function", "stream", "org", "com", "net", "junit", "jupiter", "api",
             "mockito", "hamcrest", "slf4j", "awaitility"}


def declared_names(code):
    """Names javalang knows are methods / locals. Empty sets if the snippet won't parse."""
    methods, variables = set(), set()
    try:
        tree = javalang.parse.parse("public class PH_Dummy {\n" + code + "\n}")
    except Exception:
        return methods, variables, False
    for _, n in tree.filter(javalang.tree.MethodDeclaration):
        methods.add(n.name)
    for _, n in tree.filter(javalang.tree.MethodInvocation):
        if n.member:
            methods.add(n.member)
    for t in (javalang.tree.LocalVariableDeclaration, javalang.tree.FieldDeclaration,
              javalang.tree.VariableDeclaration):
        for _, n in tree.filter(t):
            for d in getattr(n, "declarators", []) or []:
                variables.add(d.name)
    for _, n in tree.filter(javalang.tree.FormalParameter):
        variables.add(n.name)
    return methods, variables, True


def placeholderize(code):
    """Return (new_code, mapping dict, stats dict)."""
    try:
        toks = list(javalang.tokenizer.tokenize(code))
    except Exception:
        return code, {}, {"parsed": False, "tokenized": False, "replaced": 0, "identifiers": 0}

    methods, variables, parsed = declared_names(code)

    # which lines are package/import statements -> their dotted names become PKG_n
    lines = code.split("\n")
    pkg_lines = {i + 1 for i, l in enumerate(lines)
                 if re.match(r"\s*(package|import)\b", l)}

    counters = {"CLASS": 0, "METHOD": 0, "VAR": 0, "PKG": 0}
    mapping = OrderedDict()

    def placeholder(name, kind):
        if name in mapping:
            return mapping[name]
        counters[kind] += 1
        mapping[name] = f"{kind}_{counters[kind]}"
        return mapping[name]

    # collect edits as (line, col, old, new); apply right-to-left per line
    edits = []
    n_ident = 0
    for idx, t in enumerate(toks):
        if not isinstance(t, javalang.tokenizer.Identifier):
            continue
        n_ident += 1
        name = t.value
        if name in ALLOW:
            continue
        line, col = t.position
        if line in pkg_lines:
            if name in PKG_ALLOW:
                continue
            kind = "PKG"
        else:
            nxt = toks[idx + 1].value if idx + 1 < len(toks) else ""
            if name in methods or nxt == "(":
                kind = "METHOD"
            elif name[:1].isupper():
                kind = "CLASS"
            elif name in variables:
                kind = "VAR"
            else:
                kind = "VAR"
        edits.append((line, col, name, placeholder(name, kind)))

    by_line = {}
    for line, col, old, new in edits:
        by_line.setdefault(line, []).append((col, old, new))
    out = list(lines)
    for line, items in by_line.items():
        if line - 1 >= len(out):
            continue
        s = out[line - 1]
        for col, old, new in sorted(items, reverse=True):
            i = col - 1                      # javalang columns are 1-based
            if s[i:i + len(old)] == old:
                s = s[:i] + new + s[i + len(old):]
        out[line - 1] = s

    return "\n".join(out), dict(mapping), {
        "parsed": parsed, "tokenized": True,
        "replaced": len(edits), "identifiers": n_ident,
        "distinct_replaced": len(mapping)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="FlakeBench/FlakeBench_dataset.csv")
    ap.add_argument("--outdir", default="experiments/placeholder_study/data")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = list(csv.DictReader(open(args.inp)))
    out_csv = os.path.join(args.outdir, "FlakeBench_placeholder.csv")
    out_map = os.path.join(args.outdir, "mapping.jsonl")

    stats = {"tokenized": 0, "parsed": 0, "replaced": 0, "identifiers": 0,
             "distinct": 0, "unchanged": 0}
    with open(out_csv, "w", newline="") as fo, open(out_map, "w") as fm:
        w = csv.DictWriter(fo, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            new_code, mapping, st = placeholderize(r["full_code"])
            rec = dict(r)
            rec["full_code"] = new_code
            w.writerow(rec)
            fm.write(json.dumps({"id": r["id"], "project": r["project"],
                                 "category": r["category"], "test_name": r["test_name"],
                                 "stats": st, "mapping": mapping}) + "\n")
            stats["tokenized"] += st["tokenized"]
            stats["parsed"] += st.get("parsed", False)
            stats["replaced"] += st["replaced"]
            stats["identifiers"] += st["identifiers"]
            stats["distinct"] += st.get("distinct_replaced", 0)
            stats["unchanged"] += (new_code == r["full_code"])

    n = len(rows)
    print(f"rows: {n}")
    print(f"tokenized ok      : {stats['tokenized']} ({100*stats['tokenized']/n:.1f}%)")
    print(f"fully parsed (AST): {stats['parsed']} ({100*stats['parsed']/n:.1f}%)  "
          f"-> rest used the positional heuristic")
    print(f"identifier tokens : {stats['identifiers']}  ({stats['identifiers']/n:.1f} per test)")
    print(f"tokens replaced   : {stats['replaced']}  ({stats['replaced']/n:.1f} per test, "
          f"{100*stats['replaced']/max(stats['identifiers'],1):.1f}% of identifiers)")
    print(f"distinct names replaced per test: {stats['distinct']/n:.1f}")
    print(f"tests left unchanged: {stats['unchanged']}")
    print(f"\nwrote {out_csv}\nwrote {out_map}")


if __name__ == "__main__":
    sys.exit(main())
