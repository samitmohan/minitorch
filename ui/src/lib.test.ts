import { describe, expect, it } from "vitest";
import { downsample, formatNumber, labelOffsets, parseVariables } from "./lib";

describe("downsample", () => {
  it("returns short inputs untouched", () => {
    expect(downsample([1, 2, 3], 10)).toEqual([1, 2, 3]);
  });
  it("caps the point count and keeps the last row", () => {
    const rows = Array.from({ length: 10_001 }, (_, i) => i);
    const out = downsample(rows, 1000);
    expect(out.length).toBeLessThanOrEqual(1001);
    expect(out[0]).toBe(0);
    expect(out[out.length - 1]).toBe(10_000);
  });
});

describe("parseVariables", () => {
  it("reads scalars and lists", () => {
    expect(parseVariables("x=2, w=-0.5, v=[1, 2.5]")).toEqual({ x: 2, w: -0.5, v: [1, 2.5] });
  });
  it("rejects leftovers instead of dropping them", () => {
    expect(() => parseVariables("x=2, oops")).toThrow(/oops/);
    expect(() => parseVariables('v=["a"]')).toThrow(/list of numbers/);
  });
});

describe("formatNumber", () => {
  it("keeps four significant digits and groups big numbers", () => {
    expect(formatNumber(1.234567)).toBe("1.235");
    expect(formatNumber(54123.7)).toBe("54,124");
    expect(formatNumber(0.00012)).toBe("1.20e-4");
    expect(formatNumber(undefined)).toBe("-");
  });
});

describe("labelOffsets", () => {
  const ratioClose = (a: number, b: number) => a / b < 1.5;
  it("leaves far-apart labels at the default offset", () => {
    expect(labelOffsets({ a: 100, b: 10, c: 1 }, ratioClose)).toEqual({ a: 4, b: 4, c: 4 });
  });
  it("fans out labels that would overlap, larger value higher", () => {
    expect(labelOffsets({ cpu: 889, metal: 797, cuda: 100 }, ratioClose)).toEqual({ cpu: -3.5, metal: 11.5, cuda: 4 });
  });
});
