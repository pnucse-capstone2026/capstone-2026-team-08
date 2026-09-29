# ml-personalization — 상위 개인화 제어 모듈

Team VR 졸업과제 「몰입형 XR을 위한 상황 인식 기반 Adaptive Passthrough Framework」의
**상위 개인화 제어 계층**입니다. 사용자의 Passthrough 활성화 로그를 슬라이딩 윈도우로 요약하고,
Random Forest가 "최근 활성화가 불필요했을 확률"을 추정하면, 그 값으로 Passthrough ON 임계값을 보수적인 방향으로만 소폭 조정합니다.

담당: 따다소 (로그 feature 설계, RF 학습·ONNX 변환, 임계값 매핑)

## 전체 흐름

```
Quest 로그(risk-snapshot-*.jsonl)
        │  parse_risk_snapshot_log.py
        ▼
real_sessions.csv / real_events.csv        (활성화 이벤트 단위)
        │  build_features.py
        ▼
real_features.csv                          (윈도우 단위 7차원 feature + 라벨)
        │  train_model.py
        ▼
rf_personalization_real.joblib             (PC에서 오프라인 학습)
        │  convert_to_onnx.py
        ▼
rf_personalization_real.onnx
        │  convert_tree_onnx_for_unity.py
        ▼
unity-client/Assets/Models/personalization_runtime.onnx
        │  Unity InferenceEngine (온디바이스 추론)
        ▼
risk_probability → 임계값 보정치 δ → 정적/동적 위험 임계값 반영
```

- 학습은 **PC에서 오프라인**으로만 이루어지며, Quest에서는 배포된 모델의 **추론만** 수행합니다.
  (기기 안에서 자동 재학습하지 않습니다.)
- Unity 쪽 런타임은 `unity-client/Assets/Scripts/PersonalizationRuntimeController.cs`,
  임계값 수식은 `AdaptivePassthrough/Core/PersonalizationModels.cs`의 `PersonalizationMath`에 있습니다.
  연동 상세는 [`docs/ONDEVICE_SENTIS_INTEGRATION.md`](../docs/ONDEVICE_SENTIS_INTEGRATION.md)를 참고하세요.

## 동작 원리

### 1. Feature (7차원)

윈도우(연속된 활성화 이벤트 묶음) 하나당 아래 벡터 하나를 만듭니다. 컬럼 순서는 `config.py`의
`FEATURE_COLUMNS`가 기준이며 Unity의 `PersonalizationFeatureVector.ToArray()`와 동일해야 합니다.

| 순서 | feature | 의미 |
|---|---|---|
| 1 | `f_pt` | 윈도우 내 활성화 빈도 (`N_activate / W`) |
| 2 | `r_cancel` | 수동 해제 비율 |
| 3 | `t_pt_bar` | 평균 Passthrough 지속 시간(초) |
| 4 | `v_h_bar` | 평균 머리 이동 속도(m/s) |
| 5 | `v_h_max` | 최대 머리 이동 속도(m/s) |
| 6 | `A_space_norm` | 플레이 공간 면적 정규화 (기준 4~8㎡) |
| 7 | `T_session_norm` | 세션 경과 시간 정규화 (기준 30분) |

### 2. 라벨링

활성화 이벤트의 지속 시간으로 Positive / Negative / Neutral을 부여하고, 윈도우 안에서
비율이 0.4 이상인 쪽을 윈도우 라벨로 삼습니다. 임계값은 `config.py`에 있습니다.

| 데이터 | Negative (불필요) | Positive (필요) | 윈도우 크기 / stride |
|---|---|---|---|
| mock | 지속 2.0초 미만 | 지속 3.0초 이상 | 10 / 5 |
| real | 지속 0.15초 미만 | 지속 0.3초 이상 | 3 / 1 |

real 임계값은 실제 Quest 로그(활성화 평균 약 0.17초)를 보고 정한 **임시값**입니다.

### 3. 모델

- `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)`
- 실제 로그에는 Negative 표본이 없어(아래 [한계](#한계) 참고) 현재 모델의 클래스는
  `Neutral`, `Positive` 두 개이며, **`Neutral`을 위험(Negative) 쪽으로 간주**합니다.
- Unity가 `TreeEnsembleClassifier`/`ZipMap`을 읽지 못하므로, `convert_tree_onnx_for_unity.py`가
  랜덤 포레스트를 표준 ONNX 연산(`Gather`/`LessOrEqual`/`Where` 등)으로 풀어서 단일 출력
  `risk_probability = P(Neutral) = 1 - P(Positive)`를 내보냅니다. 클래스 순서가
  `['Neutral', 'Positive']`가 아니면 변환이 실패하도록 검증합니다.

### 4. 임계값 보정

```
δ = clip( (p_negative − 0.5) × 0.2 , 0 , 0.10 )
새 임계값 = min( 기본 임계값 + δ , 0.95 )
```

- `p_negative`가 0.5 이하이면 δ = 0이라 **기본값을 그대로** 씁니다. 임계값을 더 민감하게
  낮추는 방향의 조정은 하지 않습니다 (안전 우선).
- δ의 최대치는 0.10, 임계값 상한은 0.95입니다.
- 정적 임계값(`stable_on` / `rapid_on` / `hand_full`)과 동적 임계값(`dynamic_on` / `dynamic_off`)
  총 5개에 반영됩니다. `dynamic_off`는 `dynamic_on`을 넘지 않도록 히스테리시스를 유지합니다.

### 5. 안전장치

| 장치 | 내용 |
|---|---|
| Cold-start 게이트 | 누적 세션 수가 5 미만이면 개인화하지 않고 기본 임계값 사용 (`N_MIN_SESSIONS`) |
| Shadow Mode | Unity 기본값 ON. 예측은 로그로만 남기고 임계값에는 적용하지 않음 (별도로 APPLY를 켜야 적용) |
| 상향 조정만 허용 | 위 δ 규칙 (음수 δ는 0으로 clip) |
| 상한 클램프 | `MAX_THRESHOLD = 0.95` — 안전 기능이 사실상 꺼지는 것을 방지 |
| 비정상 입력 방어 | NaN/Infinity는 기본값으로 대체 (Unity `PersonalizationMath`) |

## 설치 및 실행

```bash
cd ml-personalization
pip install -r requirements.txt
cd src
```

### A. mock 데이터로 파이프라인 확인

```bash
python generate_mock_logs.py       # 1. mock 세션/이벤트 생성 (원룸 2~8㎡ 가정)
python build_features.py           # 2. 라벨링 + feature 추출
python train_model.py              # 3. RF 학습
python convert_to_onnx.py          # 4. ONNX 변환 + sklearn 결과와 클래스/확률 일치 검증
python cold_start.py               # 5. cold-start 게이트 동작 확인
python personalize.py              # 6. 임계값 보정 수식 확인
python test_personalization.py     # 7. 통합 테스트 (cold_start + personalize)
python hyperparam_tuning.py        # 8. 윈도우 크기/stride 비교 (선택)
```

mock 결과(accuracy 약 0.97)는 라벨이 feature에서 파생된 구조라서 나오는 값입니다.
**코드가 정상 동작하는지 확인하는 용도**이며 모델 성능의 근거가 아닙니다.

### B. 실제 Quest 로그로 모델 만들기

1. **로그 꺼내기**: 앱 실행 중 `RiskSnapshotSessionLogger`가 기기의
   `Application.persistentDataPath/RiskLogs/risk-snapshot-YYYYMMDD-HHmmss.jsonl`에 로그를 씁니다.
   `adb pull` 또는 Device File Explorer로 PC에 복사하세요. 세션마다 파일이 하나씩 생깁니다.

2. **CSV로 변환**
   ```bash
   python parse_risk_snapshot_log.py --input risk-snapshot-20260728-141623.jsonl --session-id s1 --space-m2 4.0
   python parse_risk_snapshot_log.py --input risk-snapshot-20260729-091000.jsonl --session-id s2 --space-m2 4.0 --append
   ```
   - `--session-id`: 세션 이름 (겹치지 않게)
   - `--space-m2`: 로그를 찍은 방의 크기(㎡). 로그에 없는 값이라 직접 입력
   - `--append`: 두 번째 파일부터 필수 (없으면 기존 CSV를 덮어씀)
   - 끝에 "활성화 이벤트 0개"가 나오면 그 구간에 Passthrough가 한 번도 켜지지 않은 것입니다.

3. **feature 추출 → 학습 → ONNX**
   ```bash
   python build_features.py --source real
   python train_model.py --source real          # 윈도우 5개 미만이면 중단 → 로그를 더 모을 것
   python convert_to_onnx.py --source real      # models/rf_personalization_real.onnx 생성
   python test_personalization.py --source real # 세션별 결과 확인
   ```

4. **Unity용 변환** (저장소 루트에서 실행)
   ```bash
   cp ml-personalization/models/rf_personalization_real.onnx \
      unity-client/Assets/Models/rf_personalization_real.onnx.source
   python ml-personalization/src/convert_tree_onnx_for_unity.py
   ```
   결과물 `unity-client/Assets/Models/personalization_runtime.onnx`를 Unity에서
   `ModelAsset`으로 임포트합니다. (Windows PowerShell에서는 `cp` 대신 `Copy-Item`을 쓰세요.)

## 파일 구성

| 파일 | 역할 |
|---|---|
| `src/config.py` | 공유 상수 (`FEATURE_COLUMNS`, 라벨링 임계값, 윈도우 크기, `N_MIN_SESSIONS`, 기본 임계값 등) |
| `src/generate_mock_logs.py` | mock 세션/이벤트 로그 생성 |
| `src/parse_risk_snapshot_log.py` | 실제 `risk-snapshot-*.jsonl`(10Hz 스냅샷)에서 활성화 구간을 뽑아 `real_*.csv`로 변환 |
| `src/build_features.py` | 라벨링 + 슬라이딩 윈도우 7차원 feature 추출 (`--source mock/real`) |
| `src/train_model.py` | RF 학습 및 평가, feature importance 출력 |
| `src/convert_to_onnx.py` | sklearn → ONNX 변환 (`zipmap=False`, opset 17 고정), 클래스/확률 일치 검증 |
| `src/convert_tree_onnx_for_unity.py` | Unity InferenceEngine 호환 ONNX로 변환, `risk_probability` 출력 |
| `src/cold_start.py` | cold-start 판단 + 최종 진입점 `get_personalized_params()` |
| `src/personalize.py` | `p_negative` → 임계값 보정 (`compute_personalized_params`) |
| `src/test_personalization.py` | cold_start + personalize 통합 확인 |
| `src/hyperparam_tuning.py` | 윈도우 크기/stride 조합 비교 |

`data/`(CSV)와 `models/`(joblib, onnx)는 스크립트 실행 시 생성됩니다.
`ml-personalization/models/*.joblib`, `*.onnx`는 `.gitignore` 대상이며,
Unity에서 실제로 쓰는 모델은 `unity-client/Assets/Models/`에 커밋되어 있습니다.

## 한계

- **Negative 표본 없음**: 현재 앱에는 사용자가 Passthrough를 직접 끄는 기능이 없어
  `is_manual_cancel`이 항상 0입니다. 그래서 수동 해제 기반 Negative 라벨이 만들어지지 않고,
  실제 학습은 사실상 Positive/Neutral 이진 판단에 가깝습니다.
- **공간 크기 수동 입력**: 방 크기(㎡)가 로그에 남지 않아 `--space-m2`로 직접 입력해야 합니다.
- **개인화 효과 미검증**: 사용자 실험(N=12)에서는 개인화 모듈을 통제 변인으로 제외했습니다.
  `ADJUSTMENT_SCALE`(0.2), real 라벨링 임계값, 윈도우 크기/stride는 임시값입니다.
- **정규화 기준**: 공간 4~8㎡, 세션 30분은 가정치이며 실제 Scene API 값 범위와의 대조가 남아 있습니다.
  초기 설계 기준이라 `test_personalization.py`의 출력 수치와 다를 수 있습니다.
- `train_model.py`, `personalize.py` 등의 주석에는 초기 설계(가중치 `w_c/w_s/w_d/w_i`, `tau_viz`)
  표현이 일부 남아 있습니다. 현재 출력은 임계값 조정 방식입니다.
