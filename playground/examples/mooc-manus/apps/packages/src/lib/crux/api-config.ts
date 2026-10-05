export const LOCAL_API_BASE_URL = "http://localhost:5150";

/** HTTP 页面通过同源 /api 代理；桌面 file 页面使用完整后端地址。 */
export function resolveApiBaseUrl({
  configured,
  location,
}: {
  configured?: string;
  location: Pick<Location, "origin" | "protocol">;
}): string {
  return (
    configured?.trim() ||
    (location.protocol === "file:" ? LOCAL_API_BASE_URL : location.origin)
  );
}
