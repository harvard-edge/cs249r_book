/**
 * NapkinCalc unit-contract guardrails.
 *
 * Two defects this file pins down, both invisible to a typecheck because
 * every value involved is a bare `number`:
 *
 *  1. FORMULAS.training_flops(params_b, tokens_b) takes tokens in BILLIONS,
 *     but the calculator's input is labeled "Tokens (T)". Passing the raw
 *     field value understated training FLOPs by 1000x.
 *  2. AllReduce was fed the GPU's HBM bandwidth (H100: 3350 GB/s) instead of
 *     the interconnect it actually crosses (NVLink 900 GB/s within a node,
 *     InfiniBand NDR 50 GB/s between nodes), making the network look 4-67x
 *     faster than any real fabric.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import NapkinCalc from '@/components/NapkinCalc';
import { FORMULAS, INTERCONNECTS } from '@/lib/hardware';

describe('NapkinCalc unit contract', () => {
  it('treats the "Tokens (T)" input as trillions, not billions', () => {
    render(<NapkinCalc defaultOpen />);
    fireEvent.click(screen.getByRole('button', { name: 'Training Time' }));

    // Defaults: 70B params, 1T tokens -> 6 * 70e9 * 1e12 = 4.2e23 FLOPs.
    // The detail line prints FLOPs in units of 1e21, so 420.0, not 0.4.
    expect(screen.getByText(/6 × 70B × 1T = 420\.0e21 FLOPS/)).toBeTruthy();

    // Scales with the input, so a hardcoded 1000 would not satisfy this.
    // The labels here are not associated with their inputs, so reach the
    // field through the label's sibling rather than getByLabelText.
    const tokensInput = screen.getByText('Tokens (T)').parentElement!.querySelector('input')!;
    fireEvent.change(tokensInput, { target: { value: '2' } });
    expect(screen.getByText(/6 × 70B × 2T = 840\.0e21 FLOPS/)).toBeTruthy();
  });

  it('sizes AllReduce by the interconnect, not by HBM bandwidth', () => {
    render(<NapkinCalc defaultOpen />);
    fireEvent.click(screen.getByRole('button', { name: 'AllReduce Time' }));

    // 64 GPUs spans multiple 8-GPU nodes, so the message crosses InfiniBand.
    const ib = INTERCONNECTS.find(i => i.name === 'InfiniBand NDR')!;
    expect(screen.getByText(new RegExp(`InfiniBand NDR @ ${ib.bandwidth_gbs} GB/s`))).toBeTruthy();

    // 140 GB of gradients over 50 GB/s is seconds, not the ~82 ms that HBM
    // bandwidth produced.
    const gradSize = FORMULAS.model_memory_gb(70, 2);
    const expected = FORMULAS.allreduce_time_ms(gradSize, ib.bandwidth_gbs, 64);
    expect(expected).toBeGreaterThan(1000);
    expect(screen.getByText(`${expected.toFixed(1)} ms`)).toBeTruthy();
  });
});
