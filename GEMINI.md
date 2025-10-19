# Whisper Hallucination Suppression via DPO (for **gemini-cli**)

> 목적: Whisper 계열 ASR에서 **비음성/잡음/긴 침묵**에 기인한 환각을 **선호 최적화(DPO)**로 억제하고, *삭제 증가* 없이 안전하게 배포 가능한 학습·평가 파이프라인을 **gemini-cli**로 재현.

---

## 0. 스코프 요약

* **입력 모달리티**: 오디오(16kHz mono) → 텍스트
* **학습 순서**: (선택) SFT → **DPO** (초기 목표: 바닐라 DPO)

  *고급 옵션(후속 연구)*: Cal‑DPO 앵커항, Mask‑DPO(스팬 표적), TDPO(토큰 표적)
* **데이터**:

  * chosen: AI-Hub 실오디오(권장) + 필요 시 CosyVoice TTS 보완
  * rejected: 현재 정책(베이스 혹은 SFT 모델) 추론 산출물에서 환각 성향 높은 출력 자동 채집
  * 증강: MUSAN/DEMAND 잡음, 무음/삽입 위치/길이/SNR 스케줄링
* **평가**: 표준 WER/CER + **HER(환각 계열)** 지표(삽입률, FPM‑NS, 루핑 비율 등)
* **인퍼런스 가드**: no_speech/logprob/compression gating + BoH 후처리(옵션)

---

## 1. 리포지토리 구조 (권장)

```
repo/
├─ gemini.md                      # 본 문서
├─ pyproject.toml | requirements.txt
├─ .env.example
├─ configs/
│   ├─ data.yaml                  # 원천 데이터/증강/VAD/샘플링 스키마
│   ├─ train_sft.yaml             # SFT(선택) 하이퍼파라미터
│   ├─ train_dpo.yaml             # DPO 하이퍼파라미터(β, γ 등)
│   └─ eval.yaml                  # 평가 셋/지표 스위치
├─ data/
│   ├─ raw/                       # 원천 오디오(AI-Hub 등)
│   ├─ tts/                       # CosyVoice 합성(선택)
│   ├─ noise/{musan, demand}/     # 공개 잡음 코퍼스
│   ├─ processed/                 # VAD/정규화/증강 산출
│   └─ dpo_triplets/              # JSONL: {audio, chosen, rejected, meta}
├─ src/
│   ├─ audio/
│   │   ├─ vad_silero.py          # Silero/WebRTC-VAD 래퍼, 스무딩/튜닝
│   │   ├─ augment.py             # SNR/삽입 위치/크로스페이드/제로-크로싱
│   │   └─ features.py            # WhisperProcessor로 mel/input_features
│   ├─ dataset/
│   │   ├─ build_triplets.py      # chosen/rejected 자동 구축 파이프라인
│   │   └─ collator_whisper.py    # 오디오 seq2seq용 콜레이터
│   ├─ modeling/
│   │   ├─ dpo_trainer_whisper.py # TRL 상속 compute_loss 커스텀(바닐라 DPO)
│   │   ├─ losses.py              # Cal‑DPO 앵커, 길이 정규화 옵션
│   │   └─ utils.py               # logprobs, length norm, mask 적용 유틸
│   ├─ eval/
│   │   ├─ metrics.py             # WER/CER/IR/DR/FPM‑NS/looping/BoH-hit
│   │   ├─ eval_standard.py       # 표준셋 회귀 점검
│   │   └─ eval_challenge.py      # 챌린지셋(HER) 점검
│   └─ inference/
│       ├─ decode_whisper.py      # 게이팅 파라미터, BoH 후처리(옵션)
│       └─ postprocess_boh.py     # Bag of Hallucinations 필터링
├─ scripts/
│   ├─ build_data.sh              # 원시→triplets 파이프라인
│   ├─ train_sft.sh               # (선택) SFT 학습
│   ├─ train_dpo.sh               # DPO 학습
│   ├─ eval_all.sh                # 전체 평가 루틴
│   └─ export_boh.sh              # 환각 상투구 집계/갱신
└─ gemini/
    ├─ tasks.yaml                 # gemini-cli 태스크 레시피
    └─ profiles.yaml              # 런타임 프로파일(GPU/노드/로깅)
```

---

## 2. 환경 & 의존성

* **Python**: 3.10+
* **핵심 라이브러리**: `torch`, `torchaudio`, `transformers`, `datasets`, `accelerate`, `trl`, `peft`, `bitsandbytes(옵션)`, `jiwer`, `librosa`, `pydub`, `audiomentations`, `soundfile`, `webrtcvad`, `numpy`, `pandas`, `hydra-core`, `tqdm`
* **모델**: `openai/whisper-{tiny|base|small|medium}` (언어/리소스에 맞게)

예시 `requirements.txt` 스니펫

```
transformers>=4.44
datasets>=2.20
accelerate>=0.33
trl>=0.9
peft>=0.11
jiwer>=3.0
librosa>=0.10
audiomentations>=0.36
pydub>=0.25
webrtcvad>=2.0
soundfile>=0.12

# CUDA 환경에 맞는 torch/torchaudio 버전 선택
```

---

## 3. 데이터셋 구축(자동)

### 3.1 원칙

* **chosen(정답)**: 가능한 **실오디오**(AI-Hub 등)의 정답 스크립트를 사용. TTS는 **보완재**로 희귀 패턴 통제 실험에 활용.
* **rejected(비선호)**: 동일 오디오(혹은 무음·잡음 삽입본)를 **현재 정책 모델**로 인식하여 수집(온폴리시).
* **환각 유도**: 문두/문말/문중에 **무음·잡음 삽입**, 다양한 **SNR** 샘플링.
* **편집 아티팩트 회피**: **크로스페이드(10–50ms) + 제로‑크로싱 컷**.

### 3.2 VAD 세팅(한국어 롱폼 권장)

* input: 16 kHz mono, `min_speech_duration_ms=200~300`, `min_silence_duration_ms=250~400`, `speech_prob_th≈0.5`에서 ROC 튜닝
* 짧은 무음(≤120ms) 병합, 호흡/구강음 스무딩

### 3.3 증강 스케줄링(예시)

* **SNR** ∈ {−5, 0, 5, 10, 15, 20} dB
* **삽입 위치**: 문두 40%, 문중 20%, 문말 40% (긴 침묵 가중)
* **삽입 길이**: {0.5, 1, 2, 4} s 분포 혼합

### 3.4 triplet 포맷(JSONL)

각 줄 하나의 샘플:

```json
{
  "audio_path": "data/processed/utt_001.wav",
  "sr": 16000,
  "language": "ko",
  "chosen": "정답 전사 텍스트",
  "rejected": "환각이 섞인 모델 출력",
  "meta": {
    "snr": 5,
    "insert_pos": "tail",
    "insert_len_s": 2.0,
    "vad": {"speech_prob_th": 0.5},
    "source": "aihub|tts",
    "policy": "whisper-small-<date>",
    "jiwer": {"wer": 0.48, "ins": 22, "del": 4, "sub": 9}
  }
}
```

### 3.5 구축 파이프라인 개요

1. **정규화**: 16kHz mono 변환 → RMS 정규화
2. **VAD 분할**: 음성/무음 타임스탬프 계산
3. **증강**: 지정 위치에 무음/잡음 삽입(크로스페이드)
4. **추론**: 현재 정책으로 인식 → 후보 rejected
5. **선별**: `jiwer`로 삽입 중심 샘플 우선(삽입률/IR 상위), 필요 시 BoH hit 가중
6. **출력**: JSONL triplets 저장

`scripts/build_data.sh` (예시)

```bash
python -m src.dataset.build_triplets \
  --cfg configs/data.yaml \
  --input data/raw \
  --noise-root data/noise \
  --out data/dpo_triplets/train.jsonl
```

---

## 4. TRL + Whisper: DPO 학습 설계

### 4.1 핵심 아이디어

* TRL의 `DPOTrainer`를 **상속**하여 **`compute_loss`**만 오버라이드:

  * 오디오 **input_features** 콜레이터(인코더) + 텍스트 **labels**(디코더)
  * 정책/레퍼런스 × {chosen, rejected} **teacher‑forcing log‑prob** 계산
  * **길이 정규화**(평균) 또는 길이 패널티 선택
  * (옵션) **Cal‑style 앵커항**으로 선호 응답 절대 가능도 하락 방지

### 4.2 콜레이터(개요)

```python
class WhisperDPOCollator:
    def __init__(self, processor, max_input_len):
        self.proc = processor
    def __call__(self, batch):
        wavs = [load_wav(b["audio_path"]) for b in batch]
        feats = self.proc.feature_extractor(
            wavs, sampling_rate=16000, return_tensors="pt", padding=True
        ).input_features
        # labels: tokenizer로 pad, -100 마스킹
        labels_pos = self.proc.tokenizer(
            [b["chosen"] for b in batch], return_tensors="pt", padding=True
        ).input_ids
        labels_neg = self.proc.tokenizer(
            [b["rejected"] for b in batch], return_tensors="pt", padding=True
        ).input_ids
        labels_pos[labels_pos == self.proc.tokenizer.pad_token_id] = -100
        labels_neg[labels_neg == self.proc.tokenizer.pad_token_id] = -100
        return {"input_features": feats, "labels_pos": labels_pos, "labels_neg": labels_neg}
```

### 4.3 로그우도 & 로스(개요)

```python
def seq_logprob(model, feats, labels):
    out = model(input_features=feats, labels=labels)
    # out.logits: [B, T, V]
    # gather label log-probs (teacher forcing)
    logp = log_softmax(out.logits, dim=-1)
    mask = (labels != -100)
    token_lp = logp[mask, labels[mask]]
    # 길이 정규화(평균) 권장
    return token_lp.view(mask.sum(dim=1)).mean(dim=1)  # [B]

class DPOWhisperTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        feats = inputs["input_features"]
        lp_pos_pi = seq_logprob(model, feats, inputs["labels_pos"])  # policy
        lp_neg_pi = seq_logprob(model, feats, inputs["labels_neg"])  
        with torch.no_grad():
            ref = self.ref_model
            lp_pos_ref = seq_logprob(ref, feats, inputs["labels_pos"])  # reference
            lp_neg_ref = seq_logprob(ref, feats, inputs["labels_neg"])  
        beta = self.args.beta  # e.g., 0.1
        d_pi = lp_pos_pi - lp_neg_pi
        d_ref = lp_pos_ref - lp_neg_ref
        logits = beta * (d_pi - d_ref)
        dpo_loss = -F.logsigmoid(logits).mean()
        # (옵션) Cal-style anchor: 선호 응답 절대 우도 하락 방지
        if getattr(self.args, "cal_anchor_gamma", 0.0) > 0:
            gamma = self.args.cal_anchor_gamma
            anchor = F.relu(lp_pos_ref - lp_pos_pi).mean()
            dpo_loss = dpo_loss + gamma * anchor
        return (dpo_loss, None) if return_outputs else dpo_loss
```

### 4.4 실행 스크립트(예시)

`scripts/train_dpo.sh`

```bash
export TOKENIZERS_PARALLELISM=false
accelerate launch -m src.modeling.dpo_trainer_whisper \
  --model_name openai/whisper-small \
  --train_file data/dpo_triplets/train.jsonl \
  --eval_file  data/dpo_triplets/val.jsonl \
  --beta 0.1 --cal_anchor_gamma 0.05 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 --num_train_epochs 2 \
  --fp16 --gradient_checkpointing --lora_r 16 --lora_alpha 32
```

---

## 5. 벤치마크 & 평가

### 5.1 테스트셋 구성

* **표준셋**: 일반 WER 회귀 감시(예: 공개 읽기 말뭉치 일부)
* **챌린지셋(HER)**: 긴 침묵/저SNR/비유창/세그먼트 경계 포함 데이터 별도 구축

### 5.2 지표 정의

* **WER/CER**: 전반 정확도
* **IR (Insertion Rate)**: 삽입 수 / 참조 단어 수
* **DR (Deletion Rate)**: 삭제 수 / 참조 단어 수
* **FPM‑NS (False Positives per Minute of Non‑Speech)**:

  * 분모 = **VAD로 측정된 비음성 길이(분)**
  * 분자 = 비음성 구간에서 생성된 단어 수(또는 토큰 수)
* **Looping ratio**: 동일 n‑gram(예: 3‑gram) 반복 비율
* **BoH hit‑rate(옵션)**: BoH 상투구 매칭율

### 5.3 평가 루틴(개요)

```python
from jiwer import compute_measures
# ... decode_with_gates(): no_speech/logprob/compression/cond_prev 설정 반영
meas = compute_measures(ref, hyp)
ir = meas["insertions"]/meas["reference_word_count"]
dr = meas["deletions"]/meas["reference_word_count"]
fpm_ns = non_speech_tokens / (non_speech_minutes + 1e-8)
```

`scripts/eval_all.sh`: 표준셋 WER + 챌린지셋 HER/IR/DR/FPM‑NS 동시 출력

---

## 6. 인퍼런스 가드 (학습 전/후 공통)

* 디코딩 파라미터: `no_speech_threshold`, `logprob_threshold`, `compression_ratio_threshold`, `condition_on_previous_text`, `suppress_tokens`
* **BoH 후처리(옵션)**: 상위 환각 상투구(EN/KO) 리스트 기반 제거/마스킹
* 로그: `no_speech_prob`, 평균 `logprob`, 압축율, 루핑 탐지 신호를 함께 저장하여 **학습적 개선 vs 휴리스틱 효과** 분리

---

## 7. gemini-cli 태스크 레시피(예시)

`gemini/tasks.yaml`

```yaml
version: 1
profiles:
  default: gpu_a100x1

tasks:
  data:build:
    cmd: bash scripts/build_data.sh
    deps: [configs/data.yaml]
    out: [data/dpo_triplets/train.jsonl]

  train:sft:
    cmd: bash scripts/train_sft.sh
    needs: [data:build]

  train:dpo:
    cmd: bash scripts/train_dpo.sh
    needs: [data:build]

  eval:all:
    cmd: bash scripts/eval_all.sh
    needs: [train:dpo]
```

실행 예시

```bash
gemini run data:build
# (선택) SFT
gemini run train:sft
# DPO
gemini run train:dpo
# 평가
gemini run eval:all
```

`gemini/profiles.yaml` (예시)

```yaml
gpu_a100x1:
  env:
    CUDA_VISIBLE_DEVICES: "0"
  resources:
    gpu: 1
    cpu: 8
    ram_gb: 64
```

---

## 8. 하이퍼파라미터 가이드

* **β (DPO 강도)**: 0.05–0.2 범위 스윕(삭제율·길이 확인)
* **γ (Cal‑anchor)**: 0.01–0.1 (선호 응답 절대 우도 하락 방지)
* **LoRA**: r=8–32, α=16–64 (메모리/속도 균형)
* **학습률**: 5e‑6 ~ 2e‑5 (LoRA 기준), 에폭 1–3

---

## 9. 리스크 & 완화

* **길이 바이어스→삭제 증가**: 길이 정규화(평균), Cal‑anchor, IR/DR 동시 모니터링
* **온폴리시 부족**: 주기적으로 정책 갱신 후 재수집(Iterative PO 루프)
* **TTS 과적합**: 실오디오 비중 70–90%, TTS 10–30% 보완
* **편집 아티팩트 학습**: 크로스페이드·제로‑크로싱 강제, SNR/삽입 위치 다양화

---

## 10. 체크리스트 (Go/No‑Go)

* [ ] VAD ROC 튜닝 완료(도메인 기준)
* [ ] JSONL triplets 품질 점검(랜덤 샘플 수동 검수)
* [ ] 표준셋 WER 회귀 없음
* [ ] 챌린지셋 **IR↓, FPM‑NS↓** (삭제율 DR↑ 금지)
* [ ] 인퍼런스 게이팅 + BoH 후처리 베이스라인 대비 **순 이득** 검증
* [ ] 실 배포 로그: no_speech_prob/logprob/압축율/루핑 메타 저장

---

### 부록 A. BoH(옵션) 구축 요약

1. 비음성/저SNR 세그먼트 대량 생성 → Whisper 추론
2. 토큰/문장 n‑gram 빈도 집계 → 상위 환각 상투구 추출(EN/KO 분리)
3. BoH.csv 생성 → 후처리 필터에 투입(전/후 비교)

### 부록 B. 데이터 라이선스 유의

* AI-Hub/노이즈 코퍼스 사용 범위 준수, 민감 정보 비식별화, 모델 재배포 시 약관 확인
