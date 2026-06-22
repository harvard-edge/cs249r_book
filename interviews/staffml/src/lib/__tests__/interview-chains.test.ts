import { getChainsForInterview, getChainsByArea, getChainEntryPoint } from "../corpus";
import { test, expect } from "vitest";

test("getChainsForInterview returns chains for cloud L3-L5", () => {
  const chains = getChainsForInterview("cloud", ["L3", "L4", "L5"]);
  expect(chains.length).toBeGreaterThan(0);
  expect(chains[0].members.length).toBeGreaterThanOrEqual(2);
  expect(chains[0].area).toBeTruthy();
  expect(chains[0].topic).toBeTruthy();
});

test("primary tier chains sort first", () => {
  const chains = getChainsForInterview("cloud", ["L3", "L4", "L5"]);
  const firstSecondaryIdx = chains.findIndex(c => c.tier === "secondary");
  if (firstSecondaryIdx > 0) {
    const allBeforePrimary = chains.slice(0, firstSecondaryIdx).every(c => c.tier === "primary");
    expect(allBeforePrimary).toBe(true);
  }
});

test("getChainsByArea returns chains for memory", () => {
  const chains = getChainsByArea("memory", "cloud");
  expect(chains.length).toBeGreaterThan(0);
  chains.forEach(c => expect(c.area).toBe("memory"));
});

test("getChainEntryPoint finds target level", () => {
  const chains = getChainsForInterview("cloud", ["L3", "L4", "L5"]);
  const chain = chains.find(c => c.members.some(m => m.level === "L4"));
  if (chain) {
    const entry = getChainEntryPoint(chain, "L4");
    expect(entry).not.toBeNull();
    expect(entry!.level).toBe("L4");
  }
});

test("getChainEntryPoint falls back to nearest lower level", () => {
  const chains = getChainsForInterview("cloud", ["L3", "L4", "L5"]);
  const chain = chains.find(c => c.members.length >= 3 && c.members[0].level === "L3");
  if (chain) {
    const entry = getChainEntryPoint(chain, "L5");
    expect(entry).not.toBeNull();
  }
});
