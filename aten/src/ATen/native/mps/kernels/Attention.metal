// Largely influeneced by
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#include <c10/metal/utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
#include <ATen/native/mps/kernels/DecodeAttention.h>
#include <ATen/native/mps/kernels/PrefillAttention.h>

// ── forward ──────────────────────────────────────────────────────────────────

template<typename T, int D>
[[kernel]] void flash_attn_fwd(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    device       T*       O   [[buffer(3)]],
    device       float*   LSE [[buffer(4)]],
    const constant uint&  qL  [[buffer(5)]],
    const constant uint&  kL  [[buffer(6)]],
    const constant uint&  gqa [[buffer(7)]],
    const constant uint&  nh  [[buffer(8)]],
    const constant float& sc  [[buffer(9)]],
    const constant bool&  ic  [[buffer(10)]],
    const constant uint4& qs  [[buffer(11)]],
    const constant uint4& ks  [[buffer(12)]],
    const constant uint4& vs  [[buffer(13)]],
    const constant uint4& os  [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;
    constexpr int BKV = (D == 64) ? 64 : 32;

    threadgroup float K_smem[BKV * D];
    threadgroup float V_smem[BKV * D];

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;
    const uint q_max   = tgid.y * BQ + BQ - 1;

    const uint b    = bh / nh;
    const uint h    = bh % nh;
    const uint kv_h = h / gqa;

    const device T* Q_ptr = Q + b * qs[0] + h    * qs[1];
    const device T* K_ptr = K + b * ks[0] + kv_h * ks[1];
    const device T* V_ptr = V + b * vs[0] + kv_h * vs[1];
    device       T* O_ptr = O + b * os[0] + h    * os[1];

    const bool valid_q = (q_row < qL);

    float q_reg[EPL];
    for (int e = 0; e < EPL; e++)
        q_reg[e] = valid_q ? float(Q_ptr[q_row * qs[2] + lane * EPL + e]) : 0.0f;

    float acc[EPL] = {};
    float m = -INFINITY, l = 0.0f;

    const uint tg_size = 32 * BQ;  // 1024

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_max) break;

        for (uint i = tid; i < (uint)(BKV * D); i += tg_size) {
            uint r = kb + i / D;
            uint d = i % D;
            bool in = (r < kL);
            K_smem[i] = in ? float(K_ptr[r * ks[2] + d]) : 0.0f;
            V_smem[i] = in ? float(V_ptr[r * vs[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool cv = !ic || (k_row <= q_row);
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < EPL; e++)
                partial += q_reg[e] * K_smem[j * D + lane * EPL + e];
            float score = cv ? (simd_sum(partial) * sc) : -INFINITY;

            float m_new = max(m, score);
            float alpha = metal::precise::exp(m - m_new);
            float p_j   = metal::precise::exp(score - m_new);
            m = m_new;
            l = l * alpha + p_j;

            for (int e = 0; e < EPL; e++)
                acc[e] = acc[e] * alpha + p_j * V_smem[j * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int e = 0; e < EPL; e++)
        O_ptr[q_row * os[2] + lane * EPL + e] = T(acc[e] * inv_l);
    if (lane == 0)
        LSE[bh * qL + q_row] = m + log(l);
}

// ── backward preprocess ───────────────────────────────────────────────────────
// D_vec[i] = rowsum(dO_i * O_i).  Pure register reduction via simd_sum().
// Grid  : (B*H, ceil(qL/BQ), 1),  TG : (32, BQ, 1)

template<typename T, int D>
[[kernel]] void flash_attn_bwd_preprocess(
    const device T*       dO  [[buffer(0)]],
    const device T*       O   [[buffer(1)]],
    device       float*   Dv  [[buffer(2)]],
    const constant uint&  qL  [[buffer(3)]],
    const constant uint&  nh  [[buffer(4)]],
    const constant uint4& dos [[buffer(5)]],
    const constant uint4& os  [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;

    if (q_row >= qL) return;

    const uint b = bh / nh;
    const uint h = bh % nh;

    const device T* dO_ptr = dO + b * dos[0] + h * dos[1];
    const device T* O_ptr  = O  + b * os[0]  + h * os[1];

    float partial = 0.0f;
    for (int e = 0; e < EPL; e++)
        partial += float(dO_ptr[q_row * dos[2] + lane * EPL + e])
                 * float(O_ptr[q_row *  os[2]  + lane * EPL + e]);

    float total = simd_sum(partial);
    if (lane == 0)
        Dv[bh * qL + q_row] = total;
}

// ── backward dQ ──────────────────────────────────────────────────────────────
// Recomputes attention weights from saved LSE; accumulates dQ.
// Same smem strategy as forward: K+V both in threadgroup memory.
// Grid  : (B*H, ceil(qL/BQ), 1),  TG : (32, BQ, 1)

template<typename T, int D>
[[kernel]] void flash_attn_bwd_dq(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    const device T*       O   [[buffer(3)]],
    const device T*       dO  [[buffer(4)]],
    const device float*   LSE [[buffer(5)]],
    const device float*   Dv  [[buffer(6)]],
    device       T*       dQ  [[buffer(7)]],
    const constant uint&  qL  [[buffer(8)]],
    const constant uint&  kL  [[buffer(9)]],
    const constant uint&  gqa [[buffer(10)]],
    const constant uint&  nh  [[buffer(11)]],
    const constant float& sc  [[buffer(12)]],
    const constant bool&  ic  [[buffer(13)]],
    const constant uint4& qs  [[buffer(14)]],
    const constant uint4& ks  [[buffer(15)]],
    const constant uint4& vs  [[buffer(16)]],
    const constant uint4& os  [[buffer(17)]],
    const constant uint4& dos [[buffer(18)]],
    const constant uint4& dqs [[buffer(19)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;
    constexpr int BKV = (D == 64) ? 64 : 32;

    threadgroup float K_smem[BKV * D];
    threadgroup float V_smem[BKV * D];

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;
    const uint q_max   = tgid.y * BQ + BQ - 1;

    const uint b    = bh / nh;
    const uint h    = bh % nh;
    const uint kv_h = h / gqa;

    const device T*  Q_ptr  = Q  + b * qs[0]  + h    * qs[1];
    const device T*  K_ptr  = K  + b * ks[0]  + kv_h * ks[1];
    const device T*  V_ptr  = V  + b * vs[0]  + kv_h * vs[1];
    const device T*  dO_ptr = dO + b * dos[0] + h    * dos[1];
    device       T*  dQ_ptr = dQ + b * dqs[0] + h    * dqs[1];

    const bool valid_q = (q_row < qL);

    float q_reg[EPL]  = {};
    float do_reg[EPL] = {};
    float dq_acc[EPL] = {};
    float lse_val = 0.0f, d_vec = 0.0f;

    if (valid_q) {
        for (int e = 0; e < EPL; e++) {
            q_reg[e]  = float(Q_ptr[q_row  * qs[2]  + lane * EPL + e]);
            do_reg[e] = float(dO_ptr[q_row * dos[2] + lane * EPL + e]);
        }
        lse_val = LSE[bh * qL + q_row];
        d_vec   = Dv[bh * qL + q_row];
    }

    const uint tg_size = 32 * BQ;

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_max) break;

        for (uint i = tid; i < (uint)(BKV * D); i += tg_size) {
            uint r = kb + i / D;
            uint d = i % D;
            bool in = (r < kL);
            K_smem[i] = in ? float(K_ptr[r * ks[2] + d]) : 0.0f;
            V_smem[i] = in ? float(V_ptr[r * vs[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool cv = !ic || (k_row <= q_row);
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < EPL; e++)
                partial += q_reg[e] * K_smem[j * D + lane * EPL + e];
            float score = cv ? (simd_sum(partial) * sc) : -INFINITY;
            float p_ij  = metal::precise::exp(score - lse_val);
            if (!valid_q) p_ij = 0.0f;

            float dov = 0.0f;
            for (int e = 0; e < EPL; e++)
                dov += do_reg[e] * V_smem[j * D + lane * EPL + e];
            float ds_ij = p_ij * (simd_sum(dov) - d_vec);

            for (int e = 0; e < EPL; e++)
                dq_acc[e] += ds_ij * K_smem[j * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;

    for (int e = 0; e < EPL; e++)
        dQ_ptr[q_row * dqs[2] + lane * EPL + e] = T(dq_acc[e] * sc);
}

// ── backward dK + dV ─────────────────────────────────────────────────────────
// K and V live in per-simdgroup registers.  Q and dO are tiled through smem.
// Grid  : (B*H, ceil(kL/BK), 1),  TG : (32, BK, 1)
//
// Smem budget: Q_smem + dO_smem = BQS * D * sizeof(float) * 2 ≤ 32 KB
// → BQS = (D == 64) ? 64 : 32

template<typename T, int D>
[[kernel]] void flash_attn_bwd_dkdv(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    const device T*       O   [[buffer(3)]],   // unused, kept for dispatch compat
    const device T*       dO  [[buffer(4)]],
    const device float*   LSE [[buffer(5)]],
    const device float*   Dv  [[buffer(6)]],
    device       T*       dK  [[buffer(7)]],
    device       T*       dV  [[buffer(8)]],
    const constant uint&  qL  [[buffer(9)]],
    const constant uint&  kL  [[buffer(10)]],
    const constant uint&  gqa [[buffer(11)]],
    const constant uint&  nh  [[buffer(12)]],
    const constant float& sc  [[buffer(13)]],
    const constant bool&  ic  [[buffer(14)]],
    const constant uint4& qs  [[buffer(15)]],
    const constant uint4& ks  [[buffer(16)]],
    const constant uint4& vs  [[buffer(17)]],
    const constant uint4& os  [[buffer(18)]],
    const constant uint4& dos [[buffer(19)]],
    const constant uint4& dks [[buffer(20)]],
    const constant uint4& dvs [[buffer(21)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL  = (D + 31) / 32;
    constexpr int BK   = 32;
    constexpr int BQS  = (D == 64) ? 64 : 32;

    threadgroup float  Q_smem[BQS * D];
    threadgroup float dO_smem[BQS * D];

    const uint lane    = tid % 32;
    const uint k_local = tid / 32;
    const uint bh      = tgid.x;
    const uint k_row   = tgid.y * BK + k_local;
    const uint k_min   = tgid.y * BK;

    // bh encodes (batch, kv_head): bh = b * nh + kv_h  (nh == kvH here)
    const uint kv_h = bh % nh;
    const uint b    = bh / nh;

    const device T*  K_ptr  = K  + b * ks[0]  + kv_h * ks[1];
    const device T*  V_ptr  = V  + b * vs[0]  + kv_h * vs[1];
    device       T*  dK_ptr = dK + b * dks[0] + kv_h * dks[1];
    device       T*  dV_ptr = dV + b * dvs[0] + kv_h * dvs[1];

    const bool valid_k = (k_row < kL);

    float k_reg[EPL] = {};
    float v_reg[EPL] = {};
    if (valid_k) {
        for (int e = 0; e < EPL; e++) {
            k_reg[e] = float(K_ptr[k_row * ks[2] + lane * EPL + e]);
            v_reg[e] = float(V_ptr[k_row * vs[2] + lane * EPL + e]);
        }
    }

    float dk_acc[EPL] = {};
    float dv_acc[EPL] = {};

    const uint tg_size = 32 * BK;

    for (uint g = 0; g < gqa; g++) {
        const uint q_head = kv_h * gqa + g;
        const device T* Q_ptr  = Q  + b * qs[0]  + q_head * qs[1];
        const device T* dO_ptr = dO + b * dos[0] + q_head * dos[1];
        const uint bh_lse = b * (nh * gqa) + q_head;

    for (uint qb = 0; qb < qL; qb += BQS) {
        if (ic && qb + (uint)BQS - 1 < k_min) continue;

        for (uint i = tid; i < (uint)(BQS * D); i += tg_size) {
            uint r = qb + i / D;
            uint d = i % D;
            bool in = (r < qL);
            Q_smem[i]  = in ? float( Q_ptr[r * qs[2]  + d]) : 0.0f;
            dO_smem[i] = in ? float(dO_ptr[r * dos[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(qb + (uint)BQS, qL);

        for (uint q_row = qb; q_row < tile_end; ++q_row) {
            if (ic && k_row > q_row) continue;

            float lse_i   = LSE[bh_lse * qL + q_row];
            float d_vec_i = Dv[bh_lse * qL + q_row];
            int i = (int)(q_row - qb);

            float qk = 0.0f;
            for (int e = 0; e < EPL; e++)
                qk += Q_smem[i * D + lane * EPL + e] * k_reg[e];
            float p_ij = metal::precise::exp(simd_sum(qk) * sc - lse_i);
            if (!valid_k) p_ij = 0.0f;

            float dov = 0.0f;
            for (int e = 0; e < EPL; e++)
                dov += dO_smem[i * D + lane * EPL + e] * v_reg[e];
            float ds_ij = p_ij * (simd_sum(dov) - d_vec_i);

            for (int e = 0; e < EPL; e++)
                dv_acc[e] += p_ij * dO_smem[i * D + lane * EPL + e];

            for (int e = 0; e < EPL; e++)
                dk_acc[e] += ds_ij * Q_smem[i * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    } // end gqa g-loop

    if (!valid_k) return;

    for (int e = 0; e < EPL; e++) {
        dK_ptr[k_row * dks[2] + lane * EPL + e] = T(dk_acc[e] * sc);
        dV_ptr[k_row * dvs[2] + lane * EPL + e] = T(dv_acc[e]);
    }
}

// ── explicit instantiation macros ────────────────────────────────────────────

#define INST_FLASH_FWD(T, D) \
  template [[host_name("flash_attn_fwd_" #T "_" #D)]] [[kernel]] \
  void flash_attn_fwd<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      device       T*        O   [[buffer(3)]],  \
      device       float*    LSE [[buffer(4)]],  \
      const constant uint&   qL  [[buffer(5)]],  \
      const constant uint&   kL  [[buffer(6)]],  \
      const constant uint&   gqa [[buffer(7)]],  \
      const constant uint&   nh  [[buffer(8)]],  \
      const constant float&  sc  [[buffer(9)]],  \
      const constant bool&   ic  [[buffer(10)]], \
      const constant uint4&  qs  [[buffer(11)]], \
      const constant uint4&  ks  [[buffer(12)]], \
      const constant uint4&  vs  [[buffer(13)]], \
      const constant uint4&  os  [[buffer(14)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_PRE(T, D) \
  template [[host_name("flash_attn_bwd_pre_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_preprocess<T, D>( \
      const device T*        dO  [[buffer(0)]], \
      const device T*        O   [[buffer(1)]], \
      device       float*    Dv  [[buffer(2)]], \
      const constant uint&   qL  [[buffer(3)]], \
      const constant uint&   nh  [[buffer(4)]], \
      const constant uint4&  dos [[buffer(5)]], \
      const constant uint4&  os  [[buffer(6)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_DQ(T, D) \
  template [[host_name("flash_attn_bwd_dq_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_dq<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      const device T*        O   [[buffer(3)]],  \
      const device T*        dO  [[buffer(4)]],  \
      const device float*    LSE [[buffer(5)]],  \
      const device float*    Dv  [[buffer(6)]],  \
      device       T*        dQ  [[buffer(7)]],  \
      const constant uint&   qL  [[buffer(8)]],  \
      const constant uint&   kL  [[buffer(9)]],  \
      const constant uint&   gqa [[buffer(10)]], \
      const constant uint&   nh  [[buffer(11)]], \
      const constant float&  sc  [[buffer(12)]], \
      const constant bool&   ic  [[buffer(13)]], \
      const constant uint4&  qs  [[buffer(14)]], \
      const constant uint4&  ks  [[buffer(15)]], \
      const constant uint4&  vs  [[buffer(16)]], \
      const constant uint4&  os  [[buffer(17)]], \
      const constant uint4&  dos [[buffer(18)]], \
      const constant uint4&  dqs [[buffer(19)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_DKDV(T, D) \
  template [[host_name("flash_attn_bwd_dkdv_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_dkdv<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      const device T*        O   [[buffer(3)]],  \
      const device T*        dO  [[buffer(4)]],  \
      const device float*    LSE [[buffer(5)]],  \
      const device float*    Dv  [[buffer(6)]],  \
      device       T*        dK  [[buffer(7)]],  \
      device       T*        dV  [[buffer(8)]],  \
      const constant uint&   qL  [[buffer(9)]],  \
      const constant uint&   kL  [[buffer(10)]], \
      const constant uint&   gqa [[buffer(11)]], \
      const constant uint&   nh  [[buffer(12)]], \
      const constant float&  sc  [[buffer(13)]], \
      const constant bool&   ic  [[buffer(14)]], \
      const constant uint4&  qs  [[buffer(15)]], \
      const constant uint4&  ks  [[buffer(16)]], \
      const constant uint4&  vs  [[buffer(17)]], \
      const constant uint4&  os  [[buffer(18)]], \
      const constant uint4&  dos [[buffer(19)]], \
      const constant uint4&  dks [[buffer(20)]], \
      const constant uint4&  dvs [[buffer(21)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_ALL(T) \
  INST_FLASH_FWD(T, 64)      \
  INST_FLASH_FWD(T, 128)     \
  INST_FLASH_BWD_PRE(T, 64)  \
  INST_FLASH_BWD_PRE(T, 128) \
  INST_FLASH_BWD_DQ(T, 64)   \
  INST_FLASH_BWD_DQ(T, 128)  \
  INST_FLASH_BWD_DKDV(T, 64) \
  INST_FLASH_BWD_DKDV(T, 128)

INST_FLASH_ALL(float)
INST_FLASH_ALL(half)
INST_FLASH_ALL(bfloat)


// ═══════════════════════════════════════════════════════════════════════════════
// Variable-length FlashAttention-2  (forward + backward)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Sequences of different lengths are packed end-to-end without padding.
//
// Layout  : Q  [H,   total_q, D]   K/V [kvH, total_k, D]
//           O  [H,   total_q, D]   LSE [H,   total_q]    (float)
//           Dv [H,   total_q]      (backward scratch, float)
//
// cu_seqlens : [B+1] cumulative token counts (int32, compatible with PyG batch.ptr)
// gqa        : H / kvH  (= 1 for standard MHA)
// wnd_left   : left window size  (-1 = unlimited; 0 = attend only to current token from left)
// wnd_right  : right window size (-1 = unlimited; 0 = causal, same as is_causal=true)
// alibi      : [H] ALiBi slopes (positive; bias = slope * position_delta, subtracted from score)
//              For causal:       delta = k_pos - q_pos  (<= 0)
//              For bidirectional: delta = -|k_pos - q_pos|  (<= 0)
//
// Grid      : (H,   ceil(max_seqlen_q / BQ), B)   for forward, preprocess, dQ
//           : (kvH, ceil(max_seqlen_k / BK), B)   for dK+dV
// Threadgroup: 1024 flat threads  (32 lanes x 32 rows)
// ═══════════════════════════════════════════════════════════════════════════════

// ── varlen forward ────────────────────────────────────────────────────────────

template<typename T, int D>
[[kernel]] void flash_attn_varlen_fwd(
    const device T*       Q            [[buffer(0)]],   // [H,   total_q, D]
    const device T*       K            [[buffer(1)]],   // [kvH, total_k, D]
    const device T*       V            [[buffer(2)]],   // [kvH, total_k, D]
    device       T*       O            [[buffer(3)]],   // [H,   total_q, D]
    device       float*   LSE          [[buffer(4)]],   // [H,   total_q]
    const device uint*    cu_seqlens_q [[buffer(5)]],   // [B+1]
    const device uint*    cu_seqlens_k [[buffer(6)]],   // [B+1]
    const constant uint&  total_q      [[buffer(7)]],
    const constant uint&  total_k      [[buffer(8)]],
    const constant float& sc           [[buffer(9)]],   // attention scale (1/sqrt(D))
    const constant bool&  ic           [[buffer(10)]],  // is_causal
    const constant uint&  gqa          [[buffer(11)]],  // H / kvH  (1 for MHA)
    const constant int&   wnd_left     [[buffer(12)]],  // left  window (-1 = unlimited)
    const constant int&   wnd_right    [[buffer(13)]],  // right window (-1 = unlimited)
    const device float*   alibi        [[buffer(14)]],  // [H] slopes, or dummy if no alibi
    const constant bool&  has_alibi    [[buffer(15)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = (D + 31) / 32;
    constexpr int BQ  = 32;
    constexpr int BKV = (D <= 64) ? 64 : (D <= 128) ? 32 : (D <= 256) ? 16 : 8;

    threadgroup float K_smem[BKV * D];
    threadgroup float V_smem[BKV * D];

    const uint h    = tgid.x;        // query head   [0, H)
    const uint b    = tgid.z;        // batch element [0, B)
    const uint kv_h = h / gqa;       // kv head      [0, kvH)
    const uint lane    = tid % 32;
    // elements valid for this lane when D is not a multiple of 32
    const int epl = min(EPL, max(0, (int)D - (int)lane * EPL));
    const uint q_local = tid / 32;

    const uint q_start = cu_seqlens_q[b];
    const uint q_end   = cu_seqlens_q[b + 1];
    const uint k_start = cu_seqlens_k[b];
    const uint k_end   = cu_seqlens_k[b + 1];
    const uint qL = q_end - q_start;
    const uint kL = k_end - k_start;

    if (tgid.y * BQ >= qL) return;

    const uint q_row     = tgid.y * BQ + q_local;
    const uint q_row_max = tgid.y * BQ + BQ - 1;   // last row in this Q-tile
    const bool valid_q   = (q_row < qL);

    const device T* Q_ptr = Q + h    * total_q * D + q_start * D;
    const device T* K_ptr = K + kv_h * total_k * D + k_start * D;
    const device T* V_ptr = V + kv_h * total_k * D + k_start * D;
    device       T* O_ptr = O + h    * total_q * D + q_start * D;

    float q_reg[EPL] = {};
    for (int e = 0; e < epl; e++)
        q_reg[e] = valid_q ? float(Q_ptr[q_row * D + lane * EPL + e]) : 0.0f;

    float acc[EPL] = {};
    float m = -INFINITY, l = 0.0f;

    const uint tg_size = 32 * BQ;

    for (uint kb = 0; kb < kL; kb += BKV) {
        // Causal early-exit: entire K-tile is past the last Q row in this tile
        if (ic && kb > q_row_max) break;
        // Window-right early-exit: entire K-tile is past window for all Q rows in this tile
        if (wnd_right >= 0 && (int)kb > (int)q_row_max + wnd_right) break;
        // Window-left skip: entire K-tile is too far left for all Q rows in this tile
        if (wnd_left >= 0 && (int)(kb + (uint)BKV - 1) + wnd_left < (int)(tgid.y * BQ)) continue;

        for (uint i = tid; i < (uint)(BKV * D); i += tg_size) {
            uint r = kb + i / D;
            uint d = i % D;
            bool in = (r < kL);
            K_smem[i] = in ? float(K_ptr[r * D + d]) : 0.0f;
            V_smem[i] = in ? float(V_ptr[r * D + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool in_causal = !ic || (k_row <= q_row);
            const bool in_wl = (wnd_left  < 0) || (int(k_row) + wnd_left  >= int(q_row));
            const bool in_wr = (wnd_right < 0) || (int(k_row) <= int(q_row) + wnd_right);
            const bool mask_ok = in_causal && in_wl && in_wr;
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < epl; e++)
                partial += q_reg[e] * K_smem[j * D + lane * EPL + e];

            float score = mask_ok ? (simd_sum(partial) * sc) : -INFINITY;
            if (has_alibi && mask_ok) {
                int rel = int(k_row) - int(q_row);
                score += alibi[h] * float(ic ? rel : -abs(rel));
            }

            float m_new = max(m, score);
            // Guard: exp(-inf - (-inf)) = exp(NaN) when no valid K seen yet.
            // When m_new==-inf, alpha must be 1 (l==0 anyway; acc unchanged).
            // When score==-inf, p_j must be 0 (masked token, no contribution).
            float alpha = (m_new == -INFINITY) ? 1.0f : metal::precise::exp(m - m_new);
            float p_j   = (score  == -INFINITY) ? 0.0f : metal::precise::exp(score - m_new);
            m = m_new;
            l = l * alpha + p_j;

            for (int e = 0; e < epl; e++)
                acc[e] = acc[e] * alpha + p_j * V_smem[j * D + lane * EPL + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int e = 0; e < epl; e++)
        O_ptr[q_row * D + lane * EPL + e] = T(acc[e] * inv_l);
    if (lane == 0)
        LSE[h * total_q + q_start + q_row] = m + metal::precise::log(l);
}

// ── varlen backward preprocess ────────────────────────────────────────────────
// Dv[h * total_q + q_start + q_row] = rowsum(dO * O)
// No GQA or window/ALiBi dependency: pure Q-side operation.
// Grid : (H, ceil(max_qL/BQ), B),  TG : 1024 flat threads

template<typename T, int D>
[[kernel]] void flash_attn_varlen_bwd_preprocess(
    const device T*       dO           [[buffer(0)]],  // [H, total_q, D]
    const device T*       O            [[buffer(1)]],  // [H, total_q, D]
    device       float*   Dv           [[buffer(2)]],  // [H, total_q]
    const device uint*    cu_seqlens_q [[buffer(3)]],  // [B+1]
    const constant uint&  total_q      [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = (D + 31) / 32;
    constexpr int BQ  = 32;

    const uint h       = tgid.x;
    const uint b       = tgid.z;
    const uint lane    = tid % 32;
    // elements valid for this lane when D is not a multiple of 32
    const int epl = min(EPL, max(0, (int)D - (int)lane * EPL));
    const uint q_local = tid / 32;

    const uint q_start = cu_seqlens_q[b];
    const uint q_end   = cu_seqlens_q[b + 1];
    const uint qL      = q_end - q_start;

    if (tgid.y * BQ >= qL) return;

    const uint q_row   = tgid.y * BQ + q_local;
    const bool valid_q = (q_row < qL);

    const device T* dO_ptr = dO + h * total_q * D + q_start * D;
    const device T* O_ptr  = O  + h * total_q * D + q_start * D;

    float partial = 0.0f;
    if (valid_q) {
        for (int e = 0; e < epl; e++)
            partial += float(dO_ptr[q_row * D + lane * EPL + e])
                     * float( O_ptr[q_row * D + lane * EPL + e]);
    }
    float total = simd_sum(partial);
    if (lane == 0 && valid_q)
        Dv[h * total_q + q_start + q_row] = total;
}

// ── varlen backward dQ ────────────────────────────────────────────────────────
// Grid : (H, ceil(max_qL/BQ), B),  TG : 1024 flat threads

template<typename T, int D>
[[kernel]] void flash_attn_varlen_bwd_dq(
    const device T*       Q            [[buffer(0)]],   // [H,   total_q, D]
    const device T*       K            [[buffer(1)]],   // [kvH, total_k, D]
    const device T*       V            [[buffer(2)]],   // [kvH, total_k, D]
    const device T*       dO           [[buffer(3)]],   // [H,   total_q, D]
    const device float*   LSE          [[buffer(4)]],   // [H,   total_q]
    const device float*   Dv           [[buffer(5)]],   // [H,   total_q]
    device       T*       dQ           [[buffer(6)]],   // [H,   total_q, D]
    const device uint*    cu_seqlens_q [[buffer(7)]],   // [B+1]
    const device uint*    cu_seqlens_k [[buffer(8)]],   // [B+1]
    const constant uint&  total_q      [[buffer(9)]],
    const constant uint&  total_k      [[buffer(10)]],
    const constant float& sc           [[buffer(11)]],  // attention scale (1/sqrt(D))
    const constant bool&  ic           [[buffer(12)]],  // is_causal
    const constant uint&  gqa          [[buffer(13)]],  // H / kvH  (1 for MHA)  // H / kvH
    const constant int&   wnd_left     [[buffer(14)]],
    const constant int&   wnd_right    [[buffer(15)]],
    const device float*   alibi        [[buffer(16)]],
    const constant bool&  has_alibi    [[buffer(17)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = (D + 31) / 32;
    constexpr int BQ  = 32;
    constexpr int BKV = (D <= 64) ? 64 : (D <= 128) ? 32 : (D <= 256) ? 16 : 8;

    threadgroup float K_smem[BKV * D];
    threadgroup float V_smem[BKV * D];

    const uint h    = tgid.x;
    const uint b    = tgid.z;
    const uint kv_h = h / gqa;
    const uint lane    = tid % 32;
    // elements valid for this lane when D is not a multiple of 32
    const int epl = min(EPL, max(0, (int)D - (int)lane * EPL));
    const uint q_local = tid / 32;

    const uint q_start = cu_seqlens_q[b];
    const uint q_end   = cu_seqlens_q[b + 1];
    const uint k_start = cu_seqlens_k[b];
    const uint k_end   = cu_seqlens_k[b + 1];
    const uint qL = q_end - q_start;
    const uint kL = k_end - k_start;

    if (tgid.y * BQ >= qL) return;

    const uint q_row   = tgid.y * BQ + q_local;
    const uint q_max   = tgid.y * BQ + BQ - 1;
    const bool valid_q = (q_row < qL);

    const device T* Q_ptr  = Q  + h    * total_q * D + q_start * D;
    const device T* K_ptr  = K  + kv_h * total_k * D + k_start * D;
    const device T* V_ptr  = V  + kv_h * total_k * D + k_start * D;
    const device T* dO_ptr = dO + h    * total_q * D + q_start * D;
    device       T* dQ_ptr = dQ + h    * total_q * D + q_start * D;

    float q_reg[EPL]  = {};
    float do_reg[EPL] = {};
    float dq_acc[EPL] = {};
    float lse_val = 0.0f, d_vec = 0.0f;

    if (valid_q) {
        for (int e = 0; e < epl; e++) {
            q_reg[e]  = float(Q_ptr[q_row * D + lane * EPL + e]);
            do_reg[e] = float(dO_ptr[q_row * D + lane * EPL + e]);
        }
        lse_val = LSE[h * total_q + q_start + q_row];
        d_vec   = Dv[h * total_q + q_start + q_row];
    }

    const uint tg_size = 32 * BQ;

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_max) break;
        if (wnd_right >= 0 && (int)kb > (int)q_max + wnd_right) break;
        if (wnd_left  >= 0 && (int)(kb + (uint)BKV - 1) + wnd_left < (int)(tgid.y * BQ)) continue;

        for (uint i = tid; i < (uint)(BKV * D); i += tg_size) {
            uint r = kb + i / D;
            uint d = i % D;
            bool in = (r < kL);
            K_smem[i] = in ? float(K_ptr[r * D + d]) : 0.0f;
            V_smem[i] = in ? float(V_ptr[r * D + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool in_causal = !ic || (k_row <= q_row);
            const bool in_wl = (wnd_left  < 0) || (int(k_row) + wnd_left  >= int(q_row));
            const bool in_wr = (wnd_right < 0) || (int(k_row) <= int(q_row) + wnd_right);
            const bool mask_ok = in_causal && in_wl && in_wr;
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < epl; e++)
                partial += q_reg[e] * K_smem[j * D + lane * EPL + e];

            float score = mask_ok ? (simd_sum(partial) * sc) : -INFINITY;
            if (has_alibi && mask_ok) {
                int rel = int(k_row) - int(q_row);
                score += alibi[h] * float(ic ? rel : -abs(rel));
            }
            float p_ij  = metal::precise::exp(score - lse_val);
            if (!valid_q || !mask_ok) p_ij = 0.0f;

            float dov = 0.0f;
            for (int e = 0; e < epl; e++)
                dov += do_reg[e] * V_smem[j * D + lane * EPL + e];
            float ds_ij = p_ij * (simd_sum(dov) - d_vec);

            for (int e = 0; e < epl; e++)
                dq_acc[e] += ds_ij * K_smem[j * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;
    for (int e = 0; e < epl; e++)
        dQ_ptr[q_row * D + lane * EPL + e] = T(dq_acc[e] * sc);
}

// ── varlen backward dK + dV ───────────────────────────────────────────────────
// K/V stay in per-simdgroup registers; Q+dO are tiled through smem.
// Grid : (kvH, ceil(max_kL/BK), B),  TG : 1024 flat threads
//
// GQA: tgid.x is the kv-head index. For each kv-head we loop over all
// gqa query heads that share it, accumulating dK and dV.
//
// Window + ALiBi: applied when recomputing p_ij from Q·K to match forward pass.

template<typename T, int D>
[[kernel]] void flash_attn_varlen_bwd_dkdv(
    const device T*       Q            [[buffer(0)]],   // [H,   total_q, D]
    const device T*       K            [[buffer(1)]],   // [kvH, total_k, D]
    const device T*       V            [[buffer(2)]],   // [kvH, total_k, D]
    const device T*       dO           [[buffer(3)]],   // [H,   total_q, D]
    const device float*   LSE          [[buffer(4)]],   // [H,   total_q]
    const device float*   Dv           [[buffer(5)]],   // [H,   total_q]
    device       T*       dK           [[buffer(6)]],   // [kvH, total_k, D]
    device       T*       dV           [[buffer(7)]],   // [kvH, total_k, D]
    const device uint*    cu_seqlens_q [[buffer(8)]],   // [B+1]
    const device uint*    cu_seqlens_k [[buffer(9)]],   // [B+1]
    const constant uint&  total_q      [[buffer(10)]],
    const constant uint&  total_k      [[buffer(11)]],
    const constant float& sc           [[buffer(12)]],  // attention scale (1/sqrt(D))
    const constant bool&  ic           [[buffer(13)]],  // is_causal
    const constant uint&  gqa          [[buffer(14)]],  // H / kvH  (1 for MHA)  // H / kvH
    const constant int&   wnd_left     [[buffer(15)]],
    const constant int&   wnd_right    [[buffer(16)]],
    const device float*   alibi        [[buffer(17)]],  // [H] slopes, or dummy
    const constant bool&  has_alibi    [[buffer(18)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL  = (D + 31) / 32;
    constexpr int BK   = 32;
    constexpr int BQS  = (D <= 64) ? 64 : (D <= 128) ? 32 : (D <= 256) ? 16 : 8;

    threadgroup float  Q_smem[BQS * D];
    threadgroup float dO_smem[BQS * D];

    const uint kv_h    = tgid.x;   // kv-head index [0, kvH)
    const uint b       = tgid.z;
    const uint lane    = tid % 32;
    // elements valid for this lane when D is not a multiple of 32
    const int epl = min(EPL, max(0, (int)D - (int)lane * EPL));
    const uint k_local = tid / 32;

    const uint q_start = cu_seqlens_q[b];
    const uint q_end   = cu_seqlens_q[b + 1];
    const uint k_start = cu_seqlens_k[b];
    const uint k_end   = cu_seqlens_k[b + 1];
    const uint qL = q_end - q_start;
    const uint kL = k_end - k_start;

    if (tgid.y * BK >= kL) return;

    const uint k_row = tgid.y * BK + k_local;
    const uint k_min = tgid.y * BK;
    const bool valid_k = (k_row < kL);

    const device T* K_ptr  = K  + kv_h * total_k * D + k_start * D;
    const device T* V_ptr  = V  + kv_h * total_k * D + k_start * D;
    device       T* dK_ptr = dK + kv_h * total_k * D + k_start * D;
    device       T* dV_ptr = dV + kv_h * total_k * D + k_start * D;

    float k_reg[EPL] = {};
    float v_reg[EPL] = {};
    if (valid_k) {
        for (int e = 0; e < epl; e++) {
            k_reg[e] = float(K_ptr[k_row * D + lane * EPL + e]);
            v_reg[e] = float(V_ptr[k_row * D + lane * EPL + e]);
        }
    }

    float dk_acc[EPL] = {};
    float dv_acc[EPL] = {};

    const uint tg_size = 32 * BK;

    for (uint g = 0; g < gqa; g++) {
        const uint q_head = kv_h * gqa + g;

        const device T* Q_ptr  = Q  + q_head * total_q * D + q_start * D;
        const device T* dO_ptr = dO + q_head * total_q * D + q_start * D;

        for (uint qb = 0; qb < qL; qb += BQS) {
            // Causal early skip: all Q rows in tile are strictly before k_min (causal: k > q masked)
            if (ic && qb + (uint)BQS - 1 < k_min) continue;
            // Window-right skip: k is too far right for all Q rows in tile
            // wnd_right: k_row <= q_row + wnd_right → q_row >= k_row - wnd_right
            // If all q in tile < k_row - wnd_right, skip
            if (wnd_right >= 0 && (int)(qb + (uint)BQS - 1) + wnd_right < (int)k_row) continue;
            // Window-left skip: k is too far left for all Q rows in tile
            // wnd_left: k_row + wnd_left >= q_row → q_row <= k_row + wnd_left
            // If all q in tile > k_row + wnd_left, skip
            if (wnd_left >= 0 && (int)qb > (int)k_row + wnd_left) continue;

            for (uint i = tid; i < (uint)(BQS * D); i += tg_size) {
                uint r = qb + i / D;
                uint d = i % D;
                bool in = (r < qL);
                Q_smem[i]  = in ? float( Q_ptr[r * D + d]) : 0.0f;
                dO_smem[i] = in ? float(dO_ptr[r * D + d]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint tile_end = min(qb + (uint)BQS, qL);

            for (uint q_row = qb; q_row < tile_end; ++q_row) {
                const bool in_causal = !ic || (k_row <= q_row);
                const bool in_wl = (wnd_left  < 0) || (int(k_row) + wnd_left  >= int(q_row));
                const bool in_wr = (wnd_right < 0) || (int(k_row) <= int(q_row) + wnd_right);
                if (!in_causal || !in_wl || !in_wr) continue;

                float lse_i   = LSE[q_head * total_q + q_start + q_row];
                float d_vec_i = Dv[q_head * total_q + q_start + q_row];
                int i = (int)(q_row - qb);

                float qk = 0.0f;
                for (int e = 0; e < epl; e++)
                    qk += Q_smem[i * D + lane * EPL + e] * k_reg[e];

                float raw_score = simd_sum(qk) * sc;
                if (has_alibi) {
                    int rel = int(k_row) - int(q_row);
                    raw_score += alibi[q_head] * float(ic ? rel : -abs(rel));
                }
                float p_ij = metal::precise::exp(raw_score - lse_i);
                if (!valid_k) p_ij = 0.0f;

                float dov = 0.0f;
                for (int e = 0; e < epl; e++)
                    dov += dO_smem[i * D + lane * EPL + e] * v_reg[e];
                float ds_ij = p_ij * (simd_sum(dov) - d_vec_i);

                for (int e = 0; e < epl; e++)
                    dv_acc[e] += p_ij * dO_smem[i * D + lane * EPL + e];
                for (int e = 0; e < epl; e++)
                    dk_acc[e] += ds_ij * Q_smem[i * D + lane * EPL + e];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } // end gqa loop

    if (!valid_k) return;
    for (int e = 0; e < epl; e++) {
        dK_ptr[k_row * D + lane * EPL + e] = T(dk_acc[e] * sc);
        dV_ptr[k_row * D + lane * EPL + e] = T(dv_acc[e]);
    }
}

// ── varlen explicit instantiation ─────────────────────────────────────────────

#define INSTANTIATE_FLASH_VARLEN_FWD(T, D) \
  template [[host_name("flash_attn_varlen_fwd_" #T "_" #D)]] [[kernel]] \
  void flash_attn_varlen_fwd<T, D>( \
      const device T*       Q            [[buffer(0)]],   \
      const device T*       K            [[buffer(1)]],   \
      const device T*       V            [[buffer(2)]],   \
      device       T*       O            [[buffer(3)]],   \
      device       float*   LSE          [[buffer(4)]],   \
      const device uint*    cu_seqlens_q [[buffer(5)]],   \
      const device uint*    cu_seqlens_k [[buffer(6)]],   \
      const constant uint&  total_q      [[buffer(7)]],   \
      const constant uint&  total_k      [[buffer(8)]],   \
      const constant float& sc           [[buffer(9)]],   \
      const constant bool&  ic           [[buffer(10)]],  \
      const constant uint&  gqa          [[buffer(11)]],  \
      const constant int&   wnd_left     [[buffer(12)]],  \
      const constant int&   wnd_right    [[buffer(13)]],  \
      const device float*   alibi        [[buffer(14)]],  \
      const constant bool&  has_alibi    [[buffer(15)]],  \
      uint3 tgid [[threadgroup_position_in_grid]],        \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, D) \
  template [[host_name("flash_attn_varlen_bwd_pre_" #T "_" #D)]] [[kernel]] \
  void flash_attn_varlen_bwd_preprocess<T, D>( \
      const device T*       dO           [[buffer(0)]],  \
      const device T*       O            [[buffer(1)]],  \
      device       float*   Dv           [[buffer(2)]],  \
      const device uint*    cu_seqlens_q [[buffer(3)]],  \
      const constant uint&  total_q      [[buffer(4)]],  \
      uint3 tgid [[threadgroup_position_in_grid]],       \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, D) \
  template [[host_name("flash_attn_varlen_bwd_dq_" #T "_" #D)]] [[kernel]] \
  void flash_attn_varlen_bwd_dq<T, D>( \
      const device T*       Q            [[buffer(0)]],   \
      const device T*       K            [[buffer(1)]],   \
      const device T*       V            [[buffer(2)]],   \
      const device T*       dO           [[buffer(3)]],   \
      const device float*   LSE          [[buffer(4)]],   \
      const device float*   Dv           [[buffer(5)]],   \
      device       T*       dQ           [[buffer(6)]],   \
      const device uint*    cu_seqlens_q [[buffer(7)]],   \
      const device uint*    cu_seqlens_k [[buffer(8)]],   \
      const constant uint&  total_q      [[buffer(9)]],   \
      const constant uint&  total_k      [[buffer(10)]],  \
      const constant float& sc           [[buffer(11)]],  \
      const constant bool&  ic           [[buffer(12)]],  \
      const constant uint&  gqa          [[buffer(13)]],  \
      const constant int&   wnd_left     [[buffer(14)]],  \
      const constant int&   wnd_right    [[buffer(15)]],  \
      const device float*   alibi        [[buffer(16)]],  \
      const constant bool&  has_alibi    [[buffer(17)]],  \
      uint3 tgid [[threadgroup_position_in_grid]],        \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, D) \
  template [[host_name("flash_attn_varlen_bwd_dkdv_" #T "_" #D)]] [[kernel]] \
  void flash_attn_varlen_bwd_dkdv<T, D>( \
      const device T*       Q            [[buffer(0)]],   \
      const device T*       K            [[buffer(1)]],   \
      const device T*       V            [[buffer(2)]],   \
      const device T*       dO           [[buffer(3)]],   \
      const device float*   LSE          [[buffer(4)]],   \
      const device float*   Dv           [[buffer(5)]],   \
      device       T*       dK           [[buffer(6)]],   \
      device       T*       dV           [[buffer(7)]],   \
      const device uint*    cu_seqlens_q [[buffer(8)]],   \
      const device uint*    cu_seqlens_k [[buffer(9)]],   \
      const constant uint&  total_q      [[buffer(10)]],  \
      const constant uint&  total_k      [[buffer(11)]],  \
      const constant float& sc           [[buffer(12)]],  \
      const constant bool&  ic           [[buffer(13)]],  \
      const constant uint&  gqa          [[buffer(14)]],  \
      const constant int&   wnd_left     [[buffer(15)]],  \
      const constant int&   wnd_right    [[buffer(16)]],  \
      const device float*   alibi        [[buffer(17)]],  \
      const constant bool&  has_alibi    [[buffer(18)]],  \
      uint3 tgid [[threadgroup_position_in_grid]],        \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INSTANTIATE_FLASH_VARLEN_ALL(T)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 32)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 48)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 64)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 80)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 96)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 112)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 128)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 160)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 192)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 224)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 256)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 320)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 384)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 448)  \
  INSTANTIATE_FLASH_VARLEN_FWD(T, 512)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 32)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 48)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 64)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 80)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 96)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 112)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 128)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 160)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 192)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 224)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 256)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 320)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 384)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 448)  \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, 512)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 32)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 48)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 64)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 80)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 96)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 112)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 128)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 160)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 192)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 224)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 256)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 320)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 384)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 448)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, 512)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 32)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 48)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 64)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 80)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 96)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 112)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 128)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 160)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 192)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 224)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 256)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 320)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 384)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 448)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, 512)

INSTANTIATE_FLASH_VARLEN_ALL(float)
INSTANTIATE_FLASH_VARLEN_ALL(half)
INSTANTIATE_FLASH_VARLEN_ALL(bfloat)
