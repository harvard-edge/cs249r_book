import type { InterviewRequestPayload, ConductorResponse, ConductorMeta } from "./interview-types";

const DEFAULT_ENDPOINT = "https://mlsysbook.ai/api/staffml-interviewer";
const ENDPOINT =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_INTERVIEWER_ENDPOINT?.replace(/\/+$/, "")) ||
  DEFAULT_ENDPOINT;

const CONDUCTOR_META_DELIMITER = "\n---CONDUCTOR_META---\n";
const TIMEOUT_MS = 20_000;

const DEFAULT_META: ConductorMeta = {
  intent: "follow_up",
  questionRef: undefined,
  chainRef: undefined,
  performanceNote: undefined,
  nextAction: null,
  areaRatings: undefined,
};

export async function sendInterviewTurn(
  payload: InterviewRequestPayload,
): Promise<ConductorResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(`${ENDPOINT}/interview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Interview API returned ${res.status}: ${body.slice(0, 200)}`);
    }

    const data = await res.json() as {
      message?: string;
      meta?: Record<string, unknown>;
      provider?: string;
      vendorLabel?: string;
      modelLabel?: string;
      privacyNote?: string;
      error?: string;
    };

    if (data.error) {
      throw new Error(`Interview API error: ${data.error}`);
    }

    const rawMessage = data.message ?? "";
    const { cleanMessage, meta } = extractMeta(rawMessage);
    const serverMeta = data.meta as ConductorMeta | undefined;

    return {
      message: cleanMessage || rawMessage,
      meta: serverMeta ?? meta,
      provider: data.provider ?? "unknown",
      vendorLabel: data.vendorLabel ?? "Unknown",
      modelLabel: data.modelLabel ?? "Unknown",
      privacyNote: data.privacyNote ?? "",
    };
  } finally {
    clearTimeout(timer);
  }
}

export function extractMeta(
  rawMessage: string,
): { cleanMessage: string; meta: ConductorMeta } {
  const delimIdx = rawMessage.indexOf(CONDUCTOR_META_DELIMITER);
  if (delimIdx >= 0) {
    const clean = rawMessage.slice(0, delimIdx).trim();
    const jsonStr = rawMessage.slice(delimIdx + CONDUCTOR_META_DELIMITER.length).trim();
    try {
      const parsed = JSON.parse(jsonStr);
      return { cleanMessage: clean, meta: { ...DEFAULT_META, ...parsed } };
    } catch {
      return { cleanMessage: clean, meta: DEFAULT_META };
    }
  }

  const jsonMatch = rawMessage.match(/\n\s*\{[\s\S]*"intent"[\s\S]*\}\s*$/);
  if (jsonMatch && jsonMatch.index !== undefined) {
    try {
      const parsed = JSON.parse(jsonMatch[0].trim());
      return {
        cleanMessage: rawMessage.slice(0, jsonMatch.index).trim(),
        meta: { ...DEFAULT_META, ...parsed },
      };
    } catch { /* fall through */ }
  }

  return { cleanMessage: rawMessage.trim(), meta: DEFAULT_META };
}
