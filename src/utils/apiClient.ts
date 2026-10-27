/**
 * Robust API Client with safe JSON parsing and Render cold-start handling.
 * Eliminates "Unexpected token '<' / <!doctype" errors when receiving HTML.
 */

export const DEFAULT_PROD_API_BASE = 'https://road-accident-detection-api.onrender.com';
export const DEFAULT_LOCAL_API_BASE = 'http://localhost:8001';

/**
 * Dynamically resolves the API base URL:
 * 1. Checks localStorage for a user-specified custom API base (allows switching in UI)
 * 2. Checks import.meta.env.VITE_API_URL
 * 3. In non-localhost environments (e.g. Vercel deployment), defaults to the live Render backend
 * 4. Otherwise defaults to http://localhost:8001
 */
export function getApiBase(): string {
  if (typeof window !== 'undefined') {
    const custom = localStorage.getItem('custom_api_base');
    if (custom && custom.trim()) {
      return custom.trim().replace(/\/+$/, '');
    }
  }
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  if (typeof window !== 'undefined' && window.location) {
    const host = window.location.hostname;
    if (host !== 'localhost' && host !== '127.0.0.1' && host !== '0.0.0.0') {
      return DEFAULT_PROD_API_BASE;
    }
  }
  return DEFAULT_LOCAL_API_BASE;
}

export function setCustomApiBase(url: string | null): void {
  if (typeof window !== 'undefined') {
    if (url && url.trim()) {
      localStorage.setItem('custom_api_base', url.trim().replace(/\/+$/, ''));
    } else {
      localStorage.removeItem('custom_api_base');
    }
  }
}

export function getWsUrl(): string {
  const envWs = import.meta.env.VITE_WS_URL;
  if (envWs && envWs.trim()) {
    return envWs.trim();
  }
  const apiBase = getApiBase();
  return `${apiBase.replace(/^http/, 'ws')}/ws/alerts`;
}

export const API_BASE = getApiBase();
export const WS_URL = getWsUrl();

export interface ApiFetchOptions extends RequestInit {
  timeoutMs?: number;
  retries?: number;
  retryDelayMs?: number;
}

export class ApiError extends Error {
  status: number;
  isServerWaking: boolean;
  friendlyMessage: string;
  errorReason: string;

  constructor(status: number, message: string, friendlyMessage: string, isServerWaking = false, errorReason = '') {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.friendlyMessage = friendlyMessage;
    this.isServerWaking = isServerWaking;
    this.errorReason = errorReason || (isServerWaking ? 'Server waking up' : message);
  }
}

/**
 * Safely parses response body as JSON.
 * If HTML or invalid text is received, checks for cold-start signatures
 * and throws a structured ApiError.
 */
export async function safeParseJson<T = any>(res: Response): Promise<T> {
  const contentType = res.headers.get('content-type') || '';
  const text = await res.text();

  const isHtml = contentType.includes('text/html') ||
                 text.trim().startsWith('<!DOCTYPE') ||
                 text.trim().startsWith('<html') ||
                 text.trim().startsWith('<');

  if (isHtml || !contentType.includes('application/json')) {
    const isWaking = res.status === 502 ||
                    res.status === 503 ||
                    res.status === 504 ||
                    res.status === 404 ||
                    text.toLowerCase().includes('spin') ||
                    text.toLowerCase().includes('waking') ||
                    text.toLowerCase().includes('render');

    throw new ApiError(
      res.status,
      'Received non-JSON response from server',
      isWaking
        ? 'Backend container is waking up from free-tier sleep (typically takes 30-50s). Stand by...'
        : `Service returned unexpected response format (${res.status}). Server may be initializing.`,
      isWaking,
      'non-JSON response'
    );
  }

  try {
    return JSON.parse(text) as T;
  } catch (err: any) {
    throw new ApiError(
      res.status,
      err.message || 'JSON parse error',
      'Received malformed data from the server. Automatically retrying...',
      true
    );
  }
}

/**
 * Fetch wrapper with timeout, cold-start retry backoff, and friendly error resolution.
 */
export async function apiFetch<T = any>(
  path: string,
  options: ApiFetchOptions = {}
): Promise<T> {
  const {
    timeoutMs = 15000,
    retries = 1,
    retryDelayMs = 2000,
    ...fetchOptions
  } = options;

  const url = path.startsWith('http') ? path : `${getApiBase()}${path.startsWith('/') ? '' : '/'}${path}`;

  let attempts = 0;
  let lastError: any = null;

  while (attempts <= retries) {
    attempts++;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(url, {
        ...fetchOptions,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!res.ok) {
        // Handle specific HTTP statuses
        if (res.status === 413) {
          throw new ApiError(413, 'Payload Too Large', 'File is too large (max 10 MB). Please choose a smaller video.');
        }
        if (res.status === 415) {
          throw new ApiError(415, 'Unsupported Media Type', 'Unsupported video format. Please upload MP4 or WebM.');
        }
        if (res.status === 503) {
          throw new ApiError(503, 'Service Busy', 'Server is processing another stream. Retrying in a moment...', true);
        }
      }

      return await safeParseJson<T>(res);
    } catch (err: any) {
      clearTimeout(timeoutId);
      lastError = err;

      const isAbort = err.name === 'AbortError';

      if (attempts <= retries) {
        const backoff = retryDelayMs * Math.pow(1.5, attempts - 1);
        await new Promise((resolve) => setTimeout(resolve, backoff));
        continue;
      }

      if (err instanceof ApiError) {
        throw err;
      }

      if (isAbort) {
        throw new ApiError(
          408,
          'Request Timeout',
          'Server response timed out. The backend container may be cold-starting.',
          true,
          'timeout'
        );
      }

      const isTypeError = err instanceof TypeError || (err.message && err.message.toLowerCase().includes('fetch'));
      let reason = 'network';
      if (isTypeError) {
        if (typeof navigator !== 'undefined' && !navigator.onLine) {
          reason = 'network';
        } else if (err.message && (err.message.toLowerCase().includes('cors') || err.message.toLowerCase().includes('origin'))) {
          reason = 'CORS';
        } else {
          reason = 'network / CORS';
        }
      } else {
        reason = err.message || 'network';
      }

      throw new ApiError(
        0,
        err.message || 'Network error',
        'Could not connect to accident detection backend. Retrying connection...',
        true,
        reason
      );
    }
  }

  throw lastError;
}

export async function fetchCameras() {
  try {
    return await apiFetch<any[]>('/cameras', { timeoutMs: 4000 });
  } catch {
    // Fall back to public cameras.json
    const res = await fetch('/cameras.json');
    if (res.ok) return await res.json();
    return [];
  }
}

export async function fetchIncidents(status?: string, cameraId?: string, page = 1, pageSize = 20) {
  const params = new URLSearchParams();
  if (status && status !== 'all') params.append('status', status);
  if (cameraId && cameraId !== 'all') params.append('camera_id', cameraId);
  params.append('page', String(page));
  params.append('page_size', String(pageSize));

  return await apiFetch<{ incidents: any[]; total: number; page: number; page_size: number }>(
    `/incidents?${params.toString()}`,
    { timeoutMs: 5000 }
  );
}

export async function patchIncidentStatus(
  id: string,
  status?: 'confirmed' | 'false_alarm',
  operatorNote?: string
) {
  return await apiFetch<any>(`/incidents/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status, operator_note: operatorNote }),
    timeoutMs: 5000,
  });
}

export async function predictVideoWithMeta(
  file: File,
  meta: {
    cameraId?: string;
    cameraName?: string;
    locationLabel?: string;
    persistenceK?: number;
    cooldownWindows?: number;
  } = {}
) {
  const formData = new FormData();
  formData.append('file', file);
  if (meta.cameraId) formData.append('camera_id', meta.cameraId);
  if (meta.cameraName) formData.append('camera_name', meta.cameraName);
  if (meta.locationLabel) formData.append('location_label', meta.locationLabel);
  formData.append('persistence_k', String(meta.persistenceK ?? 1));
  formData.append('cooldown_windows', String(meta.cooldownWindows ?? 2));

  return await apiFetch<any>('/predict', {
    method: 'POST',
    body: formData,
    timeoutMs: 30000,
  });
}
