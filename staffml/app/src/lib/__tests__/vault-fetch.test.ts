import { beforeEach, afterEach, describe, expect, test, vi } from "vitest";
import {
  vaultFetchJson,
  VaultFetchError,
  CircuitOpenError,
  __resetBreakers,
  __breakerKind,
} from "../vault-fetch";

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

function networkError(): Error {
  // fetch surfaces network failures as a TypeError.
  return new TypeError("Failed to fetch");
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  __resetBreakers();
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("vaultFetchJson", () => {
  test("returns parsed JSON on success and keeps the breaker closed", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: 1 }));
    const url = "https://a.example/manifest";
    await expect(vaultFetchJson(url, { retries: 0 })).resolves.toEqual({ ok: 1 });
    expect(__breakerKind(url)).toBe("closed");
  });

  test("sends the X-Vault-Release header", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}));
    await vaultFetchJson("https://b.example/manifest", { retries: 0 });
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-Vault-Release"]).toBeTruthy();
  });

  test("does not retry a non-retryable status (404) and throws VaultFetchError", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ error: "nope" }, 404));
    await expect(
      vaultFetchJson("https://c.example/questions/x", { retries: 2 }),
    ).rejects.toBeInstanceOf(VaultFetchError);
    expect(fetchMock).toHaveBeenCalledTimes(1); // no retries on 404
  });

  test("retries a retryable status (503) then succeeds", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({}, 503))
      .mockResolvedValueOnce(jsonResponse({ ok: 2 }));
    await expect(
      vaultFetchJson("https://d.example/q", { retries: 2 }),
    ).resolves.toEqual({ ok: 2 });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test("opens the breaker after the failure threshold and then short-circuits", async () => {
    fetchMock.mockRejectedValue(networkError());
    const url = "https://e.example/q";
    // 5 failing single-attempt calls trip the breaker (FAIL_THRESHOLD = 5).
    for (let i = 0; i < 5; i++) {
      await expect(vaultFetchJson(url, { retries: 0 })).rejects.toBeInstanceOf(TypeError);
    }
    expect(__breakerKind(url)).toBe("open");
    const callsBefore = fetchMock.mock.calls.length;
    // Next call is short-circuited without touching the network.
    await expect(vaultFetchJson(url, { retries: 0 })).rejects.toBeInstanceOf(CircuitOpenError);
    expect(fetchMock.mock.calls.length).toBe(callsBefore);
  });

  test("a caller-aborted signal rejects immediately and is not retried", async () => {
    const controller = new AbortController();
    controller.abort();
    fetchMock.mockImplementation((_url: string, init: RequestInit) => {
      if (init.signal?.aborted) {
        return Promise.reject(
          new DOMException("The operation was aborted.", "AbortError"),
        );
      }
      return Promise.resolve(jsonResponse({}));
    });
    await expect(
      vaultFetchJson("https://f.example/q", { signal: controller.signal, retries: 3 }),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).toHaveBeenCalledTimes(1); // no retries after caller abort
  });

  test("breaker state is shared per origin, not per path", async () => {
    fetchMock.mockRejectedValue(networkError());
    for (let i = 0; i < 5; i++) {
      await expect(
        vaultFetchJson(`https://g.example/questions/${i}`, { retries: 0 }),
      ).rejects.toBeInstanceOf(TypeError);
    }
    // A different path on the SAME origin is short-circuited by the shared breaker.
    await expect(
      vaultFetchJson("https://g.example/search?q=x", { retries: 0 }),
    ).rejects.toBeInstanceOf(CircuitOpenError);
  });
});
