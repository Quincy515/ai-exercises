import { useCallback, useEffect, useRef, useState } from "react";
import type { Event } from "shared_types/app.js";
import {
  eventNone,
  eventGet,
  eventIncrement,
  eventDecrement,
  eventReset,
  eventLoadState,
  eventStartWatch,
  eventConfigs,
  configsEventGetAgentConfig,
  AgentConfigViewModel,
} from "shared_types/app.js";
import { Core } from "./core.js";
import type { CruxViewModel } from "./core.js";

export const cruxEvents = {
  None: eventNone,
  Get: eventGet,
  Increment: eventIncrement,
  Decrement: eventDecrement,
  Reset: eventReset,
  LoadState: eventLoadState,
  StartWatch: eventStartWatch,
  GetAgentConfig: (baseUrl: string) =>
    eventConfigs(configsEventGetAgentConfig(baseUrl)),
} as const;

export type CruxEvents = typeof cruxEvents;

/**
 * 每次挂载在客户端创建一个 Core，以 Rust view 为业务状态来源。
 * 组件通过 dispatch(events.Increment()) 发送事件，并用 ready 控制交互。
 */
export function useCrux() {
  const [view, setView] = useState<CruxViewModel>({
    text: "",
    confirmed: false,
    agent_config: new AgentConfigViewModel(null, false, null),
  });
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const core = useRef<Core | null>(null);

  useEffect(() => {
    let active = true;
    const instance = new Core(
      (next) => {
        if (active) setView(next);
      },
      (failure) => {
        if (active) setError(failure);
      },
    );
    core.current = instance;
    setReady(false);
    setError(null);
    void instance
      .initialize()
      .then(() => {
        if (active) setReady(true);
      })
      .catch((failure: unknown) => {
        if (active)
          setError(
            failure instanceof Error ? failure : new Error(String(failure)),
          );
      });

    return () => {
      active = false;
      core.current = null;
      instance.dispose();
    };
  }, []);

  const dispatch = useCallback((event: Event) => {
    if (!core.current?.ready) return;
    setError(null);
    try {
      core.current.update(event);
    } catch (failure) {
      setError(failure instanceof Error ? failure : new Error(String(failure)));
    }
  }, []);

  return { view, events: cruxEvents, dispatch, ready, error };
}

export type { CruxViewModel };
