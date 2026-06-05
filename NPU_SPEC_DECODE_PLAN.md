# NPU Speculative Decoding Plan — N-gram Lookahead

按"先低风险/高收益、后高风险/高收益"排序的阶段计划。

目标:在 Qwen3-1.7B AOT NPU 上引入 speculative decoding(n-gram lookahead 版本),把单 token decode 速度提升 3-5×。

---

## 执行进度总览

| Phase | 状态 | 实测收益 |
|---|---|---|
| 0. 基线测量 | ✅ 完成 | baseline: 316 ms/tok (3.2 tok/s) |
| 1. 关 profiling | ✅ 完成 | **21 ms/tok (47 tok/s) → 15× 加速** |
| 2. 加 verify graph | 未开始 | — |
| 3. Runner 加载 verify | 未开始 | — |
| 4. N-gram pool | 未开始 | — |
| 5. Spec 主循环 | 未开始 | — |
| 6. KV rewind | 未开始 | — |
| 7. Benchmark | 未开始 | — |

## ⭐ Phase 0+1 实测结果(2026-05-26)

测试条件:Qwen3-1.7B LPBQ-SHA,ctx544 .bin,seq_len=32 + max_new_tokens=30,5 次重复取平均。

| 配置 | Prefill (avg ms) | Prefill tok/s | Decode (avg ms) | Decode tok/s | Decode/tok (ms) | E2E (avg ms) |
|---|---:|---:|---:|---:|---:|---:|
| **Phase 0 (profiling ON, baseline)** | 818 | 39.2 | 9,468 | 3.18 | **316** | 10,286 |
| **Phase 1 (`MLLM_QNN_PROFILE_OFF=1`)** | 48 | 670.5 | 635 | 47.24 | **21** | 683 |
| **加速倍数** | **17.0×** | 17.1× | **14.9×** | 14.9× | **14.9×** | 15.1× |

### 解读

1. **关 profiling 是巨大的免费收益** —— 之前估计 2-3× 收益,**实际是 ~15×**。
2. **NPU 真实 decode 速度 = 47 tok/s** (21 ms/tok),比 CPU 单 token decode (~14 tok/s, 73 ms/tok) 快 **3.4×**。
3. **NPU 真实 prefill 速度 = 670 tok/s**,比 CPU prefill (~40 tok/s) 快 **17×**。
4. **profiling 把 NPU 拖慢了 15 倍** —— 之前所有的 NPU vs CPU 对比都被这个污染了。

### 与之前 plan 预期对比

| | Plan 预期 | 实测 |
|---|---|---|
| Phase 1 收益 | 2-3× | **15×** |
| 关 profiling 后单 token decode | ~80-120 ms | **21 ms** |
| Phase 2-7 必要性 | 高(MLP 占 70%) | 待重新评估 |

### 结论

**关 profiling 后 NPU 已经比 CPU 快 3-4×。** 是否还继续 Phase 2-7(spec decoding)需要重新评估:
- 如果目标是"边缘部署最快 decode" → NPU 关 profiling 已经达成主要目的,Phase 2+ 还能再加 1.5-2×
- 如果目标是"对比 spec decoding 收益" → 继续 Phase 2+
- 如果开发资源紧张 → Phase 1 已经够好,可以停在这里

---

## Phase 0 — 基线测量(0.5 小时)

### 目标
确定当前 baseline,后面所有加速以此为参照。

### 步骤
1. 现有 ctx544 .bin 跑 `--seq_len 32 --max_new_tokens 30` 5 次,记下 decode time 平均值
2. 同上,加 `MLLM_QNN_PROFILE_OFF=1`,再测 5 次

### 输出
两个 baseline 数字:**带 profiling** vs **关 profiling**。后面 phase 的提速以"关 profiling"为基准。

### 预期
- 带 profiling: ~310 ms/token (已知)
- 关 profiling: 估计 ~80-120 ms/token,以实测为准

---

## Phase 1 — 关 profiling + 锁 DVFS(1 小时,零代码)

### 目标
拿走"profiling 拖后腿"的水分,看清真实瓶颈。

### 步骤
```bash
# 1. 关 profiling
export MLLM_QNN_PROFILE_OFF=1

# 2. 锁 NPU perf state(确认 HtpBurst 已生效)
# qnn_aot_cfg_1.7B.json 已配 "htp_try_best_performance": "HtpBurst"
adb shell "cat /sys/class/devfreq/*qnn*/governor" 2>/dev/null || true
```

### 验证
跑 NIAH 测试,看响应是否正确 + 计时

### 预期收益
**2-3× decode 加速**(去除 profiling 开销)

### 风险
0(只是 env var + 检查)

---

## Phase 2 — 加 verify graph 进 .bin(0.5 天)

### 目标
让 .bin 多一个 `ar_len=5` 的 decode graph 给 verify 用。

### 文件改动

**`examples/qwen3_qnn_aot/compile_sha.cpp`**
- 复制 "Model length 32"(N=32 prefill block)的代码块,改成 N=5 verify block
- 跟现有的 `ar_len=32` / `ar_len=1` 两个 trace 并列

```cpp
// 加新的 block 在 saveContext 之前
// Model length 5 (verify graph for spec decoding)
{
  int VN = 5;             // K+1
  auto sequence = mllm::Tensor::zeros({1, VN}, mllm::kInt32);
  auto causal_mask = mllm::Tensor::zeros({1, 1, VN, CL}, mllm::kUInt16);
  // ... 同 prefill block 复制(改 sequence shape 为 {1, VN})
  // past_key shape: [1, H, D, CL-VN]
  // past_value shape: [1, H, CL-VN, D]
  // trace 出 ir["model_verify"]
}
```

### 编译产物
ctx544 .bin 多一个 `context.verify` graph,大小增加 ~5%。

### 验证
```bash
./build-qnn-aot/bin/mllm-qwen3-aot-sha-c \
  -m /mnt/data/chihao/output/qwen3_1.7b.mllm \
  -c examples/qwen3_qnn_aot/config_1.7B.json \
  --aot_config examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B.json \
  --context_len 544 --ar_len 32 --spec_len 5 \
  -o /mnt/data/chihao/qwen3-1.7B-lpbq-sha-ctx544-spec5.bin
```
检查输出 `.bin` 大小 + log 显示 3 个 graphs traced。

### 预期收益
0(只是基础设施)

### 风险
- QNN tiler 可能对 VN=5 不友好。如果失败,试 VN=4 或 VN=8。
- ar_len=5 的 attention shape 比较特别,可能需要调 causal_mask 形状

### 时间
0.5 天

---

## Phase 3 — Runner 端加 verify graph 加载(0.5 天)

### 目标
让 `mllm-qwen3-aot-runner` 能调用 verify graph。

### 文件改动

**`mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp/cpp`**
- 加一个 `verify_processor_` 成员(类似 `prompt_processor_` 和 `token_generator_`)
- 在 `Runner::load()` 里加载 context.verify graph

**`mllm/backends/qnn/aot_rt/TokenGenerator.hpp/cpp`**
- 加一个 `verifyForward(tokens, cur_pos)` 方法:输入 K+1 tokens,返回 K+1 token IDs (argmax)
- 内部:bind tensors,调 `QnnGraph_execute(verify_graph)`,做 argmax

### 验证
临时写一个 test 函数:输入 `[t1, t2, t3, t4, t5]` 调 verify forward,看输出 `[v1, v2, v3, v4, v5]` 是否合理(对照普通 decode 多步的结果)。

### 预期收益
0(只是基础设施)

### 风险
- KV cache 写入逻辑:verify graph 写 5 个 slot,要确保不影响后续 decode graph 的 cache 读取

### 时间
0.5 天

---

## Phase 4 — N-gram pool 实现(0.5 天)

### 目标
runner 里维护 n-gram 词典,提供 draft 接口。

### 文件改动

**新文件 `examples/qwen3_qnn_aot/ngram_pool.hpp`**
```cpp
class NGramPool {
public:
  void update(const std::vector<int64_t>& tokens);      // 加新 token 进 pool
  std::vector<int64_t> draft(
      const std::vector<int64_t>& context, int K);      // 猜 K 个

private:
  std::unordered_map<uint64_t, std::vector<int64_t>> pool_;
  // 用 hash 加速,n=2,3,4 三个粒度
  size_t max_entries_ = 100000;  // 防止内存膨胀
};
```

**`aot_run.cpp`**
- 加 `--spec_len K` flag(默认 4)
- 创建 `NGramPool` 实例,seed 从 prompt 开始

### 验证
独立单元测试:喂一段已知文本,看 `draft()` 输出是否合理。

### 时间
0.5 天

---

## Phase 5 — Speculative decode 主循环(1 天)

### 目标
把 Phase 3 + Phase 4 串起来,实现完整 spec 流程。

### 文件改动

**`aot_run.cpp`(在调用 `runner.generate` 处替换为新逻辑)** 或 **新增 `runner.specGenerate(...)` 方法**

```cpp
while (cur_pos < target_length) {
  // 1. DRAFT
  std::vector<int64_t> drafts = pool.draft(tokens, K);

  if (drafts.empty()) {
    // 退化:普通 single-token decode
    int64_t next = decode_forward(tokens.back(), cur_pos);
    tokens.push_back(next);
    cur_pos++;
    pool.update({next});
    continue;
  }

  // 2. VERIFY
  std::vector<int64_t> verify_input = {tokens.back()};
  verify_input.insert(verify_input.end(), drafts.begin(), drafts.end());
  std::vector<int64_t> verified = verify_forward(verify_input, cur_pos);

  // 3. ACCEPT / REJECT
  int accepted = 0;
  for (int i = 0; i < (int)drafts.size(); ++i) {
    if (verified[i] == drafts[i]) {
      tokens.push_back(drafts[i]);
      accepted++;
    } else {
      tokens.push_back(verified[i]);  // 老板的修正
      accepted++;
      break;
    }
  }
  // 4. BONUS
  if (accepted == (int)drafts.size()) {
    tokens.push_back(verified.back());  // 最后一套 logits 的 argmax
    accepted++;
  }
  cur_pos += accepted;

  // 5. KV cache 处理(见 Phase 6)

  // 6. 更新 pool
  pool.update(std::vector<int64_t>(tokens.end() - accepted, tokens.end()));
}
```

### 验证
- 写完先在 `--seq_len 32` 上跑,观察 accept rate
- 跑 NIAH:输出还是 `8090293`?(spec 必须保证语义等价)

### 预期收益
**1.4-1.8× decode 加速**(取决于 accept rate)

### 风险
- Accept rate 低 → 收益小
- KV cache 状态错乱 → 输出乱码

### 时间
1 天

---

## Phase 6 — KV cache rewind(0.5 天)

### 目标
处理"部分接受"时的 KV cache 修正。

### 思路
verify graph 写了 K+1 个 slot 的 KV。如果接受了 `m` 个 (m < K+1),后面 K+1-m 个 slot 的 KV 是基于错的 input 算的,不能用。

**最简单实现**:
- `cur_pos` 设回 `prev_cur_pos + m`(下次写 KV 会覆盖错误内容)
- 但 verify 写 KV 是写到位置 [prev_cur_pos, prev_cur_pos+K+1) 这些 slot
- 下次写从 cur_pos = prev_cur_pos + m 开始,覆盖 [prev_cur_pos+m, prev_cur_pos+m+1)
- [prev_cur_pos+m+1, prev_cur_pos+K+1) 这段错的 KV 暂时还在 cache 里,但不会被读(因为 mask 把这些 slot 标为 invalid)

**关键**:确认 attention mask 在新 cur_pos 下正确(未来位置都被 mask 掉)。

### 验证
- Spec accept rate < 100% 时,输出仍正常
- 跑 NIAH:还是 `8090293`?

### 风险
- 容易漏 corner case(全拒绝、bonus 接受、cur_pos 接近 CL 上限)

### 时间
0.5 天(含调试)

---

## Phase 7 — 端到端 benchmark 与对比(0.5 天)

### 目标
量化加速,确认正确性。

### 步骤
1. 跑 NIAH 5 次,确认每次输出都包含 `8090293`(正确性)
2. 跑 5 个不同 prompt(代码 / 翻译 / 创意写作),分别测:
   - decode_time
   - tokens/sec
   - accept rate(每轮 accept 的平均 token 数)
3. 跟 Phase 1(关 profiling 但无 spec)对比

### 输出表格
```
                       关 profiling    + spec   speedup
NIAH (818 tok prefill)  XX ms/tok      XX ms    1.Xx
UCSD (17 tok prefill)   XX             XX       1.Xx
代码生成                XX             XX       1.Xx
```

---

## 总体时间表

| Phase | 工作量 | 累计提速 vs 当前 baseline (310 ms/tok) |
|---|---|---|
| 0. 基线测量 | 0.5 h | — |
| 1. 关 profiling | 1 h | 2-3× |
| 2. 加 verify graph | 0.5 day | 2-3×(没用上,持平) |
| 3. Runner 加载 verify | 0.5 day | 2-3× |
| 4. N-gram pool | 0.5 day | 2-3× |
| 5. Spec 主循环 | 1 day | **3-5×** |
| 6. KV rewind | 0.5 day | 3-5×(修正正确性) |
| 7. Benchmark | 0.5 day | — |
| **总计** | **~4 天** | **3-5×** |

---

## 风险清单

| 风险 | 概率 | 缓解 |
|---|---|---|
| QNN 不支持 ar_len=5 编译 | 低 | 试 ar_len=4 或 8;ar_len=5 是常见值 |
| KV cache rewind 写错导致输出乱码 | 中 | Phase 6 单独留时间;先写 100% accept 的简化版 |
| N-gram accept rate < 30% | 低 | Phase 5 一发现就调 K;最坏退化成 1.0× |
| Verify graph 比单 decode 慢太多(> 2×) | 低 | Profile attention 部分;K 降到 3 |
| Bin 文件变大影响 device | 低 | 加 5% 在 1.6 GB 上不显著 |

---

## 决策点

- **Phase 0-1 必做**(1 小时,零风险,2-3× 收益)
- **Phase 2-7 是 1 个完整的 spec decoding 实现**,~4 天工作量
- 如果 Phase 5 跑出来 accept rate < 30%,考虑停下来评估是否上 Medusa(更大改动但更高收益)

---

## 后续优化路径(超出本 plan 范围)

按效果排序:

| 优化 | 工作量 | 额外提速 |
|---|---|---|
| **Medusa heads**(4 个 LM heads) | 1 周训练 + 1 天集成 | 1.5-2× over Phase 7 |
| **EAGLE-2** draft head | 2-4 周训练 + 1 周集成 | 2-3× over Phase 7 |
| **Tree decoding**(多候选并发 verify) | 1 周 | 1.2-1.4× |
| **CPU 多核做 draft, NPU 做 verify pipeline** | 1 周 | 1.2-1.3× |

最佳叠加方案:Phase 1+2+3+4+5+6+7 + Medusa,总计可达 5-7× decode 加速。

---

## 实施建议

1. 先做 Phase 0+1,1 小时拿到关 profiling 的真实数字
2. 如果关 profiling 后 NPU 仍然比 CPU 慢,**暂停**——重新评估是否值得继续上 spec
3. 如果 NPU 不带 profiling 已经接近或超过 CPU,继续 Phase 2-7

不要在没确认 Phase 1 收益之前就开始 Phase 2+。
