import { describe, it, expect } from 'vitest';
import { safeParseJson, ApiError } from './apiClient';

describe('API Client & Safe JSON Parsing', () => {
  it('parses valid application/json responses successfully', async () => {
    const mockResponse = new Response(JSON.stringify({ status: 'ok', data: [1, 2, 3] }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });

    const parsed = await safeParseJson(mockResponse);
    expect(parsed).toEqual({ status: 'ok', data: [1, 2, 3] });
  });

  it('detects HTML response (Render cold start) and throws friendly ApiError without raw parse error', async () => {
    const htmlBody = '<!DOCTYPE html><html><body>Render Spinning Up</body></html>';
    const makeResponse = () =>
      new Response(htmlBody, {
        status: 502,
        headers: { 'content-type': 'text/html' },
      });

    await expect(safeParseJson(makeResponse())).rejects.toThrow(ApiError);

    try {
      await safeParseJson(makeResponse());
    } catch (err: any) {
      expect(err).toBeInstanceOf(ApiError);
      expect(err.friendlyMessage).toContain('waking up');
      expect(err.message).not.toContain("Unexpected token '<'");
      expect(err.isServerWaking).toBe(true);
    }
  });

  it('detects HTML error pages even if content-type header is missing', async () => {
    const htmlBody = '<html><body>504 Gateway Timeout</body></html>';
    const mockResponse = new Response(htmlBody, { status: 504 });

    try {
      await safeParseJson(mockResponse);
    } catch (err: any) {
      expect(err).toBeInstanceOf(ApiError);
      expect(err.isServerWaking).toBe(true);
    }
  });
});
