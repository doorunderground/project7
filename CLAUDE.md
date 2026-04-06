# RC카 자율주행 프로젝트

## 프로젝트 개요
라즈베리파이 기반 RC카가 트랙 위의 차선(노란 외곽선 + 흰색 중앙 점선)을 인식하고
자율주행하는 시스템을 구현한다.

트랙 특징:
- 검정 아스팔트 위에 노란 실선(외곽)과 흰색 점선(중앙)으로 구성
- S자 커브, 직선 구간 혼합, RC카가 겨우 지나갈 정도로 좁음
- YAHBOOM 브랜드 트랙 매트

> **주의: 프로젝트 경로가 변경되었습니다**
> 기존: `C:/PROJECT77777777777777/` → 현재: `C:/PROJECT7/`

---

## 구현 방향

### 해결 전략: 다단계 우선순위 제어
1. **차선 세그멘테이션 (YOLOv8n-seg)** → 차선 이탈 감지 / 안전장치
2. **모방 학습 (MobileNetV3-Small)** → 기본 주행 제어값 출력
3. **장애물 감지 (YOLOv8n Detection)** → 장애물 회피
4. **횡단보도 감지** → 정지

**우선순위 (높음 → 낮음):**
```
횡단보도 정지 > 장애물 회피 > 차선 이탈 보정 > 모방학습
```

---

## 단계별 구현 현황

### STEP 1 - 차선 세그멘테이션 학습 [완료 ✅]
- 학습 스크립트: `C:/PROJECT7/lane_train.py`
- 데이터: `C:/PROJECT7/lane_dataset/` (LabelMe JSON → YOLO seg 자동 변환, train/valid 자동 분리)
- 모델: YOLOv8n-seg
- 클래스:
  - `inline`    (class 0): 중앙 흰색 점선
  - `outline`   (class 1): 외곽 노란 실선
  - `crosswalk` (class 2): 횡단보도 (v3에서 추가)
- 학습 결과:
  - v1: `C:/PROJECT7/runs/lane_seg_v1/weights/best.pt` (Mask mAP50=0.932)
  - v3: `C:/PROJECT7/runs/lane_seg_v3/weights/best.pt` (현재 사용 중, crosswalk 클래스 추가)
- 자율주행 코드에서 사용: `LANE_MODEL = "C:/PROJECT7/runs/lane_seg_v3/weights/best.pt"`

### STEP 2 - 모방 학습 데이터 수집 [완료 ✅]
- PC: `Pc_server(wasd).py` / 라즈베리파이: `Pi_drive(wasd).py`
- UDP 통신 (라즈베리파이 → PC: 영상, PC → 라즈베리파이: 모터 명령)
- R키로 녹화 토글, WASD(동시 입력 지원)로 조종
- 저장 형식: `frame_000001.jpg` + `frame_000001.txt` (모터값, 키 조합)
- **stop 프레임은 저장 안 함** → 클래스 불균형 방지
- 수집 데이터: `C:/PROJECT7/collected_data/` (1차: 6666장)

### STEP 3 - 모방 학습 모델 학습 [완료 ✅]
- 학습 스크립트: `C:/PROJECT7/1train_wasd.py` (또는 `1train_wasd2.py`)
- 모델: MobileNetV3-Small (ImageNet pretrained → fine-tune)
- 입력: 224×224 RGB / 출력: 키 조합 분류 (w, a, d, wa, wd 등)
- 저장: `C:/PROJECT7/drive_best2.pth` (현재 사용 중)
- 자율주행 코드에서 사용: `IMITATION_MODEL = "C:/PROJECT7/drive_best2.pth"`

### STEP 4 - 장애물 감지 모델 학습 [진행 중 🔄]
- 학습 스크립트: `C:/PROJECT7/object_train.py`
- 모델: YOLOv8n Detection
- 데이터: `C:/PROJECT7/object/20260406_181559/` (LabelMe rectangle JSON)
  - 포맷: LabelMe JSON, shape_type="rectangle"
  - 클래스: `"object"` → ID 0 (단일 클래스)
  - 이미지 크기: 640×480
  - JSON이 없는 jpg는 자동 제외
- 학습 결과: `C:/PROJECT7/runs/object_det_v1/weights/best.pt`
- 자동 복사: `C:/PROJECT7/abcde.pt` (자율주행 코드에서 사용)

  **추가 촬영분 데이터 합산 방법:**
  ```python
  DATA_DIRS = [
      ROOT_DIR / "object" / "20260406_181559",
      ROOT_DIR / "object" / "새폴더명",   # 추가
  ]
  ```

### STEP 5 - 통합 자율주행 테스트 [구현 완료 ✅]
- 파일: `C:/PROJECT7/test_auto_drive(lane)3.py` (PC에서 실행)
- 기능: 모방학습 + 차선 안전장치 + 장애물 회피 + 횡단보도 정지
- 키 조작: SPACE=일시정지/재개, L=차선안전장치토글, ESC=종료

---

## 주요 파일 구조
```
C:/PROJECT7/
├── CLAUDE.md                          ← 이 파일
├── lane_train.py                      ← STEP 1: 차선 세그멘테이션 학습 (PC)
├── 1train_wasd.py                     ← STEP 3: 모방 학습 훈련 (PC)
├── 1train_wasd2.py                    ← STEP 3: 모방 학습 훈련 v2 (PC)
├── object_train.py                    ← STEP 4: 장애물 감지 모델 학습 (PC)
├── Pc_server(wasd).py                 ← STEP 2: 데이터 수집 서버 (PC)
├── Pi_drive(wasd).py                  ← STEP 2: 라즈베리파이 클라이언트
├── test_auto_drive(lane)3.py          ← STEP 5: 통합 자율주행 (현재 최신)
├── test_auto_drive(lane).py           ← 이전 버전
├── test_auto_drive(lane)2.py          ← 이전 버전
├── test_auto_drive(wasd).py           ← 모방학습 단독 테스트
├── drive_best2.pth                    ← 학습된 모방학습 모델 (현재 사용)
├── abcde.pt                           ← 장애물 감지 모델 (object_train.py 결과 자동 복사)
├── lane_dataset/                      ← 차선 학습 데이터 (LabelMe JSON)
│   ├── train/images/
│   ├── train/labels/
│   ├── valid/images/
│   └── valid/labels/
├── object/                            ← 장애물 감지 학습 원본 데이터
│   └── 20260406_181559/               ← jpg + LabelMe JSON 혼재
├── object_dataset/                    ← object_train.py 실행 시 자동 생성
│   ├── train/images/
│   └── valid/images/
├── collected_data/                    ← 모방학습 데이터 (6666장)
└── runs/
    ├── lane_seg_v1/weights/best.pt    ← 차선 모델 v1 (Mask mAP50=0.932)
    ├── lane_seg_v3/weights/best.pt    ← 차선 모델 v3 (현재 사용, crosswalk 추가)
    └── object_det_v1/weights/best.pt  ← 장애물 모델 (학습 후 생성)
```

---

## 하드웨어
- RC카 본체: YAHBOOM 계열
- 컴퓨팅: 라즈베리파이
- 카메라: Picamera2 (640×480, RGB888)
- 학습용 PC: Windows 11, GPU 탑재

---

## 모터 설정 (라즈베리파이 GPIO)
```python
motor_a = Motor(forward=18, backward=17)   # 왼쪽 모터
motor_b = Motor(forward=22, backward=27)   # 오른쪽 모터
```

모터 명령 형식: `"left,right"` (각각 -100 ~ 100)

WASD 키 → 모터 매핑:
```
W      → (-80,  -80)   전진
S      → ( 80,   80)   후진
A      → ( 20, -100)   좌회전 (제자리)
D      → (-100,  20)   우회전 (제자리)
W + A  → (-80,  -20)   전진+좌
W + D  → (-20,  -80)   전진+우
S + A  → ( 80,   20)   후진+좌
S + D  → ( 20,   80)   후진+우
```

---

## 네트워크 설정
- 라즈베리파이 → PC: UDP 포트 5002 (카메라 프레임)
- PC → 라즈베리파이: UDP 포트 5001 (모터 명령)
- 서버 IP: 192.168.0.6 (변경 시 `Pi_drive(wasd).py`의 SERVER_IP 수정)
- 프레임 전송: 4바이트 크기 헤더 + JPEG (quality=35), 수직 flip 적용
- PC 수신 후 좌우 flip 적용 (MIRROR_LR=True)

---

## test_auto_drive(lane)3.py 핵심 파라미터

### 장애물 회피 설정
```python
OBJECT_CONF_THR       = 0.50   # 감지 최소 confidence
OBJECT_MIN_BOX_AREA   = 2000   # 최소 bbox 면적(px²) - 먼 물체 무시
OBJECT_STEER_SEC      = 1.0    # 회피 조향 유지 시간 (초) ← 0.5에서 증가
OBJECT_COOLDOWN_SEC   = 4.0    # 회피 후 재감지 방지 쿨다운
OBJECT_AVOID_SCALE    = 0.55   # 회피 강도 (약하게 → 차선 유지 범위 안)
OBJECT_CONFIRM_WINDOW = 5      # 판단 기준 프레임 수
OBJECT_CONFIRM_RATIO  = 0.6    # 5프레임 중 3번 이상 감지 시 회피 시작
```

**프레임 버퍼 감지 방식** (단일 프레임 오감지 방지):
- 매 프레임 감지 결과를 버퍼에 쌓음
- 최근 5프레임 중 60% 이상 감지 시에만 회피 트리거
- 회피 방향은 감지된 프레임들의 다수결로 결정
- 트리거 후 버퍼 초기화 → 중복 트리거 방지

**튜닝 가이드:**
- 반응이 느리다 → WINDOW 줄이거나 RATIO 낮추기
- 오감지가 많다 → RATIO 높이기 (0.8)
- 회피가 짧다 → OBJECT_STEER_SEC 늘리기

### 횡단보도 설정
```python
CROSSWALK_STOP_SEC     = 3.0   # 정지 시간
CROSSWALK_COOLDOWN_SEC = 30.0  # 재감지 방지 쿨다운
CROSSWALK_CONF_THR     = 0.45  # 감지 최소 confidence
```

### 차선 안전장치 설정
```python
LANE_THRESHOLD      = 130   # 이탈 판단 기준 (px, 중앙에서의 거리)
DEVIATION_COUNT     = 3     # N프레임 연속 이탈 시 보정 시작
CORRECTION_COOLDOWN = 0.5   # 보정 후 재보정까지 최소 간격
```

---

## lane_train.py 핵심 설계 결정
- **LabelMe JSON → YOLO seg 자동 변환**: `convert_json_to_yolo()` 함수
- **train/valid 자동 분리**: `split_dataset()` (valid ratio=0.2)
- **data.yaml 절대경로 사용** (상대경로 쓰면 오류)
- **optimizer=AdamW**, LR0=0.001, BATCH=16, workers=0 (Windows 필수)
- **mosaic=1.0**: 소규모 데이터 과적합 방지

## object_train.py 핵심 설계 결정
- **LabelMe rectangle JSON → YOLO detection txt 자동 변환**
  - 두 포인트(좌상단/우하단) → `cx cy w h` 정규화
- **task="detect"** (seg 아님) → 자율주행 코드가 boxes만 사용
- **CLASS_MAP**: `"object"` → 0, `"obstacle"` → 0
- **best.pt → abcde.pt 자동 복사** (자율주행 코드 경로 맞춤)

---

## 차선 detection이 잘 안 될 때 - fine-tuning 요청 방법

### Claude에게 요청할 내용
```
새 데이터를 추가했어. 기존 best.pt에서 fine-tuning 해줘.
- 기존 best.pt: C:/PROJECT7/runs/lane_seg_v3/weights/best.pt
- 새 결과 이름: lane_seg_v4
- LR=0.0001, epoch=50
```

### lane_train.py에서 바꿀 설정값
```python
PRETRAINED_MODEL = ROOT_DIR / "runs/lane_seg_v3/weights/best.pt"
RUN_NAME = "lane_seg_v4"   # 반드시 변경 (덮어쓰기 방지)
```
CLI: `python lane_train.py --epochs 50 --lr0 0.0001 --run-name lane_seg_v4`

**핵심 주의사항:**
- 기존 데이터 + 새 데이터 합쳐서 학습 (Catastrophic Forgetting 방지)
- RUN_NAME 반드시 변경

---

## 트러블슈팅 기록
| 문제 | 원인 | 해결 |
|------|------|------|
| 학습 시 경로 오류 | 잘못된 경로 | `C:/PROJECT7/`로 통일 |
| data.yaml 이미지 못 찾음 | 상대경로 오류 | 절대경로로 변경 |
| valid 폴더 없어서 학습 실패 | Roboflow 데이터에 valid 폴더 없음 | `split_dataset()` 추가 |
| stop 데이터 과다 | 매 프레임 저장 시 90%가 stop | `cur_key != 'stop'` 조건 추가 |
| W+A 동시 입력 안 됨 | cv2.waitKey()는 단일 키만 감지 | 키별 만료시각 독립 추적으로 해결 |
| A/D 방향 반대 | 모터 배선 방향 문제 | A↔D 모터값 스왑 |
| 회전 각도 부족 | 안쪽 모터값 40으로 회전 부족 | 안쪽 모터값 40 → 20으로 수정 |
| OMP 중복 오류 | Windows + conda 환경 충돌 | `KMP_DUPLICATE_LIB_OK=TRUE` 추가 |
| 장애물 오감지 빈번 | 단일 프레임 감지 | 프레임 버퍼 비율 방식으로 변경 (5프레임 중 60%) |
| 장애물 회피 거리 부족 | OBJECT_STEER_SEC=0.5 너무 짧음 | 1.0초로 증가 (추가 튜닝 필요) |
