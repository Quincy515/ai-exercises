import assert from "node:assert/strict";
import test from "node:test";
import { resolveApiBaseUrl } from "../dist/crux-tests/api-config.js";

test("API origin supports dev hosts, production Web, packaged Electron and overrides", () => {
  const location = { origin: "https://manus.example.com", protocol: "https:" };
  assert.equal(
    resolveApiBaseUrl({
      location: { origin: "http://localhost:3000", protocol: "http:" },
    }),
    "http://localhost:3000",
  );
  assert.equal(resolveApiBaseUrl({ location }), location.origin);
  assert.equal(
    resolveApiBaseUrl({ location: { origin: "null", protocol: "file:" } }),
    "http://localhost:5150",
  );
  assert.equal(
    resolveApiBaseUrl({ configured: " https://api.example.com ", location }),
    "https://api.example.com",
  );
  assert.equal(
    resolveApiBaseUrl({ configured: "  ", location }),
    location.origin,
  );
});
