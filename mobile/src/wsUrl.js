export function toWebSocketUrl(baseUrl, path = '/ws/imu') {
  const wsBase = baseUrl.replace(/^http/, 'ws');
  return `${wsBase}${path.startsWith('/') ? path : `/${path}`}`;
}
