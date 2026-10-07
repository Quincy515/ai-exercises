import assert from "node:assert/strict";
import test from "node:test";
import app from "shared_types/app.js";
import { request } from "../dist/crux-tests/lib/crux/key-value.js";
import { Time } from "../dist/crux-tests/lib/crux/time.js";

const { Duration, Instant, TimerId } = app;
const namespace = "mooc-manus:crux:";

function memoryStorage() {
  const data = new Map();
  return {
    get length() {
      return data.size;
    },
    key(index) {
      return [...data.keys()][index] ?? null;
    },
    getItem(key) {
      return data.get(key) ?? null;
    },
    setItem(key, value) {
      data.set(key, value);
    },
    removeItem(key) {
      data.delete(key);
    },
  };
}

test("KV preserves binary bytes, previous values, and empty values", () => {
  const storage = memoryStorage();
  const bytes = [0, 255, 128, 195, 40];
  assert.deepEqual(request({ kind: "Get", key: "binary" }, storage), {
    kind: "Ok",
    response: { kind: "Get", value: { kind: "None" } },
  });
  assert.deepEqual(
    request({ kind: "Set", key: "binary", value: bytes }, storage),
    {
      kind: "Ok",
      response: { kind: "Set", previous: { kind: "None" } },
    },
  );
  assert.equal(storage.getItem(namespace + "binary"), JSON.stringify(bytes));
  assert.deepEqual(
    request({ kind: "Get", key: "binary" }, storage).response.value,
    {
      kind: "Bytes",
      value: bytes,
    },
  );
  assert.deepEqual(
    request({ kind: "Set", key: "binary", value: [] }, storage).response
      .previous,
    {
      kind: "Bytes",
      value: bytes,
    },
  );
  assert.deepEqual(
    request({ kind: "Get", key: "binary" }, storage).response.value,
    {
      kind: "Bytes",
      value: [],
    },
  );
  assert.equal(
    request({ kind: "Exists", key: "binary" }, storage).response.is_present,
    true,
  );
  assert.deepEqual(
    request({ kind: "Delete", key: "binary" }, storage).response.previous,
    {
      kind: "Bytes",
      value: [],
    },
  );
  assert.equal(
    request({ kind: "Exists", key: "binary" }, storage).response.is_present,
    false,
  );
  assert.deepEqual(
    request({ kind: "Delete", key: "binary" }, storage).response.previous,
    { kind: "None" },
  );
});

test("KV scopes and sorts listed keys and rejects unknown cursors", () => {
  const storage = memoryStorage();
  storage.setItem("other:key", "unrelated");
  storage.setItem("mooc-manus:other:key", "unrelated");
  for (const key of ["list:b", "elsewhere", "list:a"]) {
    request({ kind: "Set", key, value: [] }, storage);
  }
  assert.deepEqual(
    request({ kind: "ListKeys", prefix: "list:", cursor: 0n }, storage),
    {
      kind: "Ok",
      response: {
        kind: "ListKeys",
        keys: ["list:a", "list:b"],
        next_cursor: 0n,
      },
    },
  );
  assert.deepEqual(
    request({ kind: "ListKeys", prefix: "", cursor: 0n }, storage).response
      .keys,
    ["elsewhere", "list:a", "list:b"],
  );
  assert.deepEqual(
    request({ kind: "ListKeys", prefix: "", cursor: 1n }, storage),
    {
      kind: "Err",
      error: { kind: "CursorNotFound" },
    },
  );
  assert.equal(storage.getItem("other:key"), "unrelated");
});

test("KV turns storage failures and invalid stored bytes into Io errors", () => {
  const blocked = {
    getItem() {
      throw new Error("Storage blocked");
    },
  };
  assert.deepEqual(request({ kind: "Get", key: "state" }, blocked), {
    kind: "Err",
    error: { kind: "Io", message: "Storage blocked" },
  });
  const storage = memoryStorage();
  for (const invalid of ["broken json", "null", "[256]", "[1.5]", '["1"]']) {
    storage.setItem(namespace + "state", invalid);
    const result = request({ kind: "Get", key: "state" }, storage);
    assert.equal(result.kind, "Err");
    assert.equal(result.error.kind, "Io");
  }
  storage.setItem = () => {
    throw new Error("Quota exceeded");
  };
  assert.deepEqual(request({ kind: "Set", key: "new", value: [1] }, storage), {
    kind: "Err",
    error: { kind: "Io", message: "Quota exceeded" },
  });
});

test("KV accesses default storage lazily and catches a blocked getter", (t) => {
  const descriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    "localStorage",
  );
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() {
      throw new Error("Storage access denied");
    },
  });
  t.after(() => {
    if (descriptor)
      Object.defineProperty(globalThis, "localStorage", descriptor);
    else delete globalThis.localStorage;
  });
  assert.deepEqual(request({ kind: "Get", key: "state" }), {
    kind: "Err",
    error: { kind: "Io", message: "Storage access denied" },
  });
});

function clock(t, initial = 1000) {
  let now = initial;
  let nextId = 0;
  const scheduled = new Map();
  t.mock.method(Date, "now", () => now);
  t.mock.method(globalThis, "setTimeout", (callback, delay) => {
    const id = ++nextId;
    scheduled.set(id, { callback, delay, deadline: now + delay });
    return id;
  });
  t.mock.method(globalThis, "clearTimeout", (id) => scheduled.delete(id));
  return {
    scheduled,
    tick(milliseconds) {
      now += milliseconds;
      const pending = [...scheduled];
      for (const [id, timer] of pending) {
        if (timer.deadline <= now && scheduled.delete(id)) timer.callback();
      }
    },
  };
}

test("Time Now converts milliseconds to Unix seconds and nanoseconds", (t) => {
  clock(t, 1_672_531_200_123);
  const time = new Time();
  const responses = [];
  time.request({ kind: "Now" }, (response) => responses.push(response));
  assert.deepEqual(responses, [
    { kind: "Now", instant: new Instant(1_672_531_200n, 123_000_000) },
  ]);
  time.dispose();
});

test("Time supports duration, absolute deadlines, and bigint timer IDs", (t) => {
  const fake = clock(t);
  const time = new Time();
  const after = new TimerId(9_007_199_254_740_993n);
  const at = new TimerId(2n);
  const responses = [];
  time.request(
    { kind: "NotifyAfter", id: after, duration: new Duration(1_000_001n) },
    (r) => responses.push(r),
  );
  assert.equal([...fake.scheduled.values()][0].delay, 2);
  fake.tick(1);
  assert.deepEqual(responses, []);
  fake.tick(1);
  assert.deepEqual(responses, [{ kind: "DurationElapsed", id: after }]);
  time.request(
    { kind: "NotifyAt", id: at, instant: new Instant(1n, 4_000_001) },
    (r) => responses.push(r),
  );
  assert.equal([...fake.scheduled.values()][0].delay, 3);
  fake.tick(3);
  assert.deepEqual(responses[1], { kind: "InstantArrived", id: at });
  assert.equal(fake.scheduled.size, 0);
  time.dispose();
});

test("Time Clear only answers the clear request and dispose cancels timers", (t) => {
  const fake = clock(t);
  const time = new Time();
  const id = new TimerId(1n);
  const elapsed = [];
  const cleared = [];
  time.request(
    { kind: "NotifyAfter", id, duration: new Duration(10_000_000n) },
    (r) => elapsed.push(r),
  );
  time.request({ kind: "Clear", id }, (r) => cleared.push(r));
  fake.tick(10);
  assert.deepEqual(elapsed, []);
  assert.deepEqual(cleared, [{ kind: "Cleared", id }]);
  time.request(
    { kind: "NotifyAfter", id, duration: new Duration(10_000_000n) },
    (r) => elapsed.push(r),
  );
  time.dispose();
  time.dispose();
  assert.equal(fake.scheduled.size, 0);
  fake.tick(10);
  time.request({ kind: "Now" }, (r) => elapsed.push(r));
  assert.deepEqual(elapsed, []);
});

test("Time splits delays beyond the browser timer limit", (t) => {
  const fake = clock(t);
  const time = new Time();
  const id = new TimerId(1n);
  const limit = 2_147_483_647;
  const responses = [];
  time.request(
    {
      kind: "NotifyAfter",
      id,
      duration: new Duration((BigInt(limit) + 5n) * 1_000_000n),
    },
    (r) => responses.push(r),
  );
  assert.equal([...fake.scheduled.values()][0].delay, limit);
  fake.tick(limit);
  assert.deepEqual(responses, []);
  assert.equal([...fake.scheduled.values()][0].delay, 5);
  fake.tick(5);
  assert.deepEqual(responses, [{ kind: "DurationElapsed", id }]);
  time.dispose();
});

test("Time instances isolate timers with the same ID", (t) => {
  const fake = clock(t);
  const first = new Time();
  const second = new Time();
  const id = new TimerId(1n);
  const responses = [];
  const operation = {
    kind: "NotifyAfter",
    id,
    duration: new Duration(1_000_000n),
  };
  first.request(operation, () => responses.push("first"));
  second.request(operation, () => responses.push("second"));
  first.dispose();
  fake.tick(1);
  assert.deepEqual(responses, ["second"]);
  second.dispose();
});
