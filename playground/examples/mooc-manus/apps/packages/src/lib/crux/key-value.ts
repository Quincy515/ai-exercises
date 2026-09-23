import {
  keyValueErrorCursorNotFound,
  keyValueErrorIo,
  keyValueResponseDelete,
  keyValueResponseExists,
  keyValueResponseGet,
  keyValueResponseListKeys,
  keyValueResponseSet,
  keyValueResultErr,
  keyValueResultOk,
  valueBytes,
  valueNone,
  type KeyValueOperation,
  type KeyValueResult,
  type Value,
} from "shared_types/app.js";

const namespace = "mooc-manus:crux:";

function read(storage: Storage, key: string): Value {
  const stored = storage.getItem(namespace + key);
  if (stored === null) return valueNone();

  const bytes: unknown = JSON.parse(stored);
  if (
    !Array.isArray(bytes) ||
    !bytes.every((byte) => Number.isInteger(byte) && byte >= 0 && byte <= 255)
  ) {
    throw new Error("Stored value is not a byte array");
  }
  return valueBytes(bytes);
}

export function request(
  operation: KeyValueOperation,
  storage?: Storage,
): KeyValueResult {
  try {
    if (operation.kind === "ListKeys" && operation.cursor !== 0n) {
      return keyValueResultErr(keyValueErrorCursorNotFound());
    }

    // 访问存储放在请求内，以便处理浏览器隐私限制产生的异常。
    const store = storage ?? globalThis.localStorage;
    switch (operation.kind) {
      case "Get":
        return keyValueResultOk(
          keyValueResponseGet(read(store, operation.key)),
        );
      case "Set": {
        const previous = read(store, operation.key);
        store.setItem(
          namespace + operation.key,
          JSON.stringify(operation.value),
        );
        return keyValueResultOk(keyValueResponseSet(previous));
      }
      case "Delete": {
        const previous = read(store, operation.key);
        store.removeItem(namespace + operation.key);
        return keyValueResultOk(keyValueResponseDelete(previous));
      }
      case "Exists":
        return keyValueResultOk(
          keyValueResponseExists(
            store.getItem(namespace + operation.key) !== null,
          ),
        );
      case "ListKeys": {
        const keys: string[] = [];
        for (let index = 0; index < store.length; index++) {
          const key = store.key(index);
          if (key?.startsWith(namespace + operation.prefix)) {
            keys.push(key.slice(namespace.length));
          }
        }
        return keyValueResultOk(keyValueResponseListKeys(keys.sort(), 0n));
      }
    }
  } catch (error) {
    return keyValueResultErr(
      keyValueErrorIo(error instanceof Error ? error.message : String(error)),
    );
  }
}
