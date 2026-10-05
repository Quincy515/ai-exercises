import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { CoreFfi, initialized } from "shared";
import app from "shared_types/app.js";
import bincode from "shared_types/bincode/index.js";

const {
  AgentConfigViewModel,
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

test("installed BoltFFI runtime matches the workspace version", () => {
  const manifest = readFileSync(
    new URL("../../Cargo.toml", import.meta.url),
    "utf8",
  );
  const expected = manifest.match(/^boltffi\s*=\s*"=([^"]+)"/m)?.[1];
  assert.ok(expected, "The workspace must pin an exact BoltFFI version");
  const shared = JSON.parse(
    readFileSync(
      new URL("package.json", import.meta.resolve("shared")),
      "utf8",
    ),
  );
  const runtime = JSON.parse(
    readFileSync(
      new URL("../package.json", import.meta.resolve("@boltffi/runtime")),
      "utf8",
    ),
  );
  assert.equal(shared.dependencies["@boltffi/runtime"], expected);
  assert.equal(runtime.version, expected);
});

test("installed bindings include their source maps and mapped sources", () => {
  const entry = import.meta.resolve("shared");
  for (const name of ["shared", "shared_node"]) {
    const js = new URL(`${name}.js`, entry);
    const reference = readFileSync(js, "utf8").match(/sourceMappingURL=(.+)/);
    assert.ok(reference, `${name}.js should reference its source map`);
    const mapUrl = new URL(reference[1].trim(), js);
    const map = JSON.parse(readFileSync(mapUrl, "utf8"));
    map.sources.forEach((source, index) => {
      assert.ok(
        typeof map.sourcesContent?.[index] === "string" ||
          existsSync(new URL(`${map.sourceRoot ?? ""}${source}`, mapUrl)),
        `Missing source for ${name}: ${source}`,
      );
    });
  }
});

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
    assert.deepEqual(
      view(core),
      new ViewModel(
        "0 (pending)",
        false,
        new AgentConfigViewModel(null, false, null),
      ),
    );

    const reset = requests(core.update(serialize(eventReset())));
    assert.deepEqual(
      reset.map(({ effect }) => effect.kind),
      ["Render"],
    );
    assert.deepEqual(
      view(core),
      new ViewModel(
        "0 (pending)",
        false,
        new AgentConfigViewModel(null, false, null),
      ),
    );

    const increment = requests(core.update(serialize(eventIncrement())));
    assert.deepEqual(increment.map(({ effect }) => effect.kind).sort(), [
      "Http",
      "Render",
    ]);
    const http = increment.find(({ effect }) => effect.kind === "Http").effect
      .value;
    assert.equal(http.method, "POST");
    assert.equal(http.url, "https://crux-counter.fly.dev/inc");
    assert.deepEqual(
      view(core),
      new ViewModel(
        "1 (pending)",
        false,
        new AgentConfigViewModel(null, false, null),
      ),
    );

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
    assert.deepEqual(
      view(core),
      new ViewModel(
        "1 (pending)",
        false,
        new AgentConfigViewModel(null, false, null),
      ),
    );
  } finally {
    core.dispose();
  }
});
