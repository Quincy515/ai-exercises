#!/usr/bin/env node

export function trackRouteHandler(pending, handler, onFailure) {
  const task = Promise.resolve().then(handler).catch(onFailure);
  pending.add(task);
  task.then(() => pending.delete(task), () => pending.delete(task));
  return task;
}

// Called after context.close while the route guard stays registered. New tasks
// already dispatched during close join the same Set before this loop completes.
export async function settleRouteHandlers(pending) {
  while (pending.size) await Promise.allSettled([...pending]);
}

export function applyFinalSafetyGate(result) {
  const violations = [];
  if (result.blockedWrites.length) violations.push('Blocked request observed during drive or cleanup');
  if (result.pageErrors.length) violations.push('Page or route handler error observed during drive or cleanup');
  if (result.postOutcomes.some(entry => entry.outcome !== 'response-received')) {
    violations.push('Authorized POST outcome remains unknown after cleanup');
    if (result.configCleanup) {
      result.configCleanup.decision = 'outcome-unknown';
      delete result.configCleanup.restored;
      delete result.configCleanup.visibleListRestored;
      if (Object.hasOwn(result.configCleanup, 'manualCleanupRequired')) result.configCleanup.manualCleanupRequired = true;
      result.cleanupError ??= 'POST outcome unknown after route handlers settled';
    }
  }
  result.safetyGate = { checkedAfterContextClose: true, violations };
  if (violations.length) {
    result.status = 'failed';
    result.reason ??= violations.join('; ');
  }
  return result;
}
