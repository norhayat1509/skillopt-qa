# skillopt-qa

[Microsoft SkillOpt](https://github.com/microsoft/SkillOpt) 的精簡、忠實重現版,
針對 **HotpotQA** 多跳推理問答任務。

SkillOpt 是一種「**文字空間優化器**」:它不去微調模型權重,而是為一個**凍結的
LLM agent** 訓練出一份可重複使用的自然語言「**技能**」(`best_skill.md`)。
整個技能學習流程仿照神經網路訓練——有 epoch、mini-batch,以及一道
**驗證閘門(validation gate)**:只有當保留集(held-out)準確率提升時,才接受該次
編輯。訓練結束後唯一保留的產物就是 `best_skill.md`,它可跨模型、跨執行環境部署。

> 論文:《SkillOpt: Executive Strategy for Self-Evolving Agent Skills》
> ([arXiv:2605.23904](https://arxiv.org/abs/2605.23904))。
> 本 repo 是獨立、簡化的教學用實作,**並非**官方程式碼。

## 運作原理

```
種子技能 ──► [讓 agent 帶著技能跑一個 train batch] ──► 軌跡(答對/答錯)
                          │
                          ▼
              [optimizer LLM 提出一次「有界」編輯]
                          │
                          ▼
              [在驗證集上評估候選技能]
                          │
              有進步? ──是──► 接受,成為新的最佳技能
                          │
                          └─否──► 拒絕,並回饋給 optimizer(別再提同樣的)
```

optimizer 實作了論文中的穩定機制:**足夠證據**(同時呈現成功與失敗案例)、
**有界文字更新**(小而精準的編輯,非整篇重寫)、**拒絕編輯記憶**
(記住被拒絕過的版本)、以及**慢更新**(每步只做一次、且須通過驗證閘門的編輯)。

## 環境需求

- Python ≥ 3.10、[`uv`](https://docs.astral.sh/uv/)
- 一個 **OpenAI 相容**的 chat endpoint。本 repo 不安裝任何 GPU 相關套件——把
  `model.base_url` 指向任何 endpoint 即可。

預設的 `configs/hotpotqa/default.yaml` 對接的是本機 **Qwen** vLLM 服務
(模型 id 為 `qwen3.6-27b`),透過 `http://localhost:8080/v1` 存取。請依你的環境
調整 `base_url`/`target_model`。Qwen3 是**推理(reasoning)模型**,因此
`max_tokens` 設得較寬鬆,以保留思考過程與最終答案所需的 token;agent 只讀取
最終的 `content`,不會讀思考內容。

## 安裝

```bash
cd skillopt-qa
uv venv
uv pip install -e ".[dev]"
cp .env.example .env        # 設定 OPENAI_API_KEY(多數本機服務任意值即可)
```

確認你的服務以及它回報的模型 id:

```bash
curl http://localhost:8080/v1/models
```

把 `configs/hotpotqa/default.yaml` 裡的 `model.target_model` /
`model.optimizer_model` 設成**完全一致**的 id。

## 1. 下載資料集

```bash
uv run skillopt-download --out data/hotpotqa --n-train 64 --n-val 64 --n-test 200
```

會從 HuggingFace 下載 HotpotQA,並寫出:

```
data/hotpotqa/
├── train/items.json
├── val/items.json
└── test/items.json
```

每筆資料格式:`{"id", "question", "context", "answers": [...]}`。train/val 取自
HotpotQA 的 `train` 切分;test 取自 `validation`(其答案為公開),因此 test 集在
優化過程中始終未被看過。

## 2. 訓練技能

```bash
uv run skillopt-train \
    --config configs/hotpotqa/default.yaml \
    --split-dir data/hotpotqa \
    --out-root outputs
```

常用覆寫參數:`--target-model`、`--base-url`、`--num-epochs`、`--batch-size`、
`--val-size`、`--workers`、`--metric {em,f1}`、`--no-test`。

產出會放在 `outputs/<run_name>/`:

```
outputs/<run_name>/
├── best_skill.md          # 可部署的產物
├── history.json           # 每步的 train/val 指標與接受/拒絕紀錄
├── skills/skill_vXXXX_*.md
├── steps/step_XXXX.json
└── test_result.json       # 在保留 test 集上的最終 EM / F1
```

## 3. 部署

`best_skill.md` 就只是一段文字。把它接到任何 QA agent 的 system prompt 前面即可
(見 `skillopt/agent.py` 的 `BASE_INSTRUCTIONS`)——甚至可以套到別的模型上。

## 專案結構

| 路徑 | 角色 |
|------|------|
| `skillopt/config.py` | YAML 設定 + CLI 覆寫 |
| `skillopt/llm.py` | OpenAI 相容的 chat 客戶端 |
| `skillopt/data.py` | HotpotQA 下載與切分建立 |
| `skillopt/agent.py` | 凍結的 QA agent + 平行 rollout |
| `skillopt/evaluator.py` | EM / token-F1 評分 |
| `skillopt/optimizer.py` | 文字空間技能優化器 |
| `skillopt/trainer.py` | epoch/batch 迴圈 + 驗證閘門 |
| `scripts/` | `download_data.py`、`train.py` 兩支 CLI |
| `tests/` | 單元測試 + 離線 e2e 測試(完全不需網路) |

## 測試

```bash
uv run pytest            # 所有測試以 fake LLM 離線執行
```

## 授權

MIT
