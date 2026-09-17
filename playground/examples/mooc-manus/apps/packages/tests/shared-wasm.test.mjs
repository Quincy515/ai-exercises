import assert from "node:assert/strict";
import test from "node:test";
import { CoreFfi, initialized } from "shared";
import app from "shared_types/app.js";
import bincode from "shared_types/bincode/index.js";

const {
  Requests,
  ViewModel,
  eventIncrement,
  eventLoadState,
  eventReset,
  keyValueResponseGet,
  keyValueResultOk,
  serializeEvent,
  serializeKeyValueResult,
  valueNone,
} = app;
const { BincodeDeserializer, BincodeSerializer } = bincode;

function serialize(event) {
  const serializer = new BincodeSerializer();
  serializeEvent(event, serializer);
  return serializer.getBytes();
}

function requests(bytes) {
  return Requests.deserialize(new BincodeDeserializer(bytes)).value;
}

function view(core) {
  return ViewModel.deserialize(new BincodeDeserializer(core.view()));
}

// A single sequential test owns the core and the generated module's WASM state.
test("generated WASM packages serialize events, decode effects, and resolve local state", async () => {
  await initialized;
  const core = CoreFfi.new({ processEffects: requests });

  try {
    assert.deepEqual(view(core), new ViewModel("0 (pending)", false));

    const reset = requests(core.update(serialize(eventReset())));
    assert.deepEqual(
      reset.map(({ effect }) => effect.kind),
      ["Render"],
    );
    assert.deepEqual(view(core), new ViewModel("0 (pending)", false));

    const increment = requests(core.update(serialize(eventIncrement())));
    assert.deepEqual(increment.map(({ effect }) => effect.kind).sort(), [
      "Http",
      "Render",
    ]);
    const http = increment.find(({ effect }) => effect.kind === "Http").effect
      .value;
    assert.equal(http.method, "POST");
    assert.equal(http.url, "https://crux-counter.fly.dev/inc");
    assert.deepEqual(view(core), new ViewModel("1 (pending)", false));

    const load = requests(core.update(serialize(eventLoadState())));
    assert.equal(load.length, 1);
    assert.deepEqual(load[0].effect, {
      kind: "KeyValue",
      value: { kind: "Get", key: "state" },
    });

    const response = new BincodeSerializer();
    serializeKeyValueResult(
      keyValueResultOk(keyValueResponseGet(valueNone())),
      response,
    );
    const resolved = requests(core.resolve(load[0].id, response.getBytes()));
    assert.deepEqual(
      resolved.map(({ effect }) => effect.kind),
      ["Render"],
    );
    assert.deepEqual(view(core), new ViewModel("1 (pending)", false));
  } finally {
    core.dispose();
  }
});
