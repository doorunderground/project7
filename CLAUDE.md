# RC카 자율주행 프로젝트

## 프로젝트 개요
라즈베리파이 기반 RC카가 트랙 위의 차선(노란 외곽선 + 흰색 중앙 점선)을 인식하고
자율주행하는 시스템을 구현한다.

트랙 특징:
- 검정 아스팔트 위에 노란 실선(외곽)과 흰색 점선(중앙)으로 구성
- S자 커브, 직선 구간 혼합, RC카가 겨우 지나갈 정도로 좁음
- YAHBOOM 브랜드 트랙 매트

---

## 구현 방향

### 해결 전략: 다단계 우선순위 제어
1. **모방 학습 (MobileNetV3-Small)** → 기본 주행 제어값 출력
2. **차선 세그멘테이션 (YOLOv8n-seg)** → 차선 이탈 감지 / 안전장치
3. **횡단보도 감지** → 정지 후 재출발
4. **장애물 감지 (YOLOv8n Detection)** → 순차 회피 기동

**우선순위 (높음 → 낮음):**
```
횡단보도 정지 > 장애물 회피 5단계 기동 > 차선 이탈 보정 > 모방학습
```

---

## 단계별 구현 현황

### STEP 1 - 모방 학습 데이터 수집 [완료 ✅]
- PC: `Pc_server(wasd).py` / 라즈베리파이: `Pi_drive(wasd).py`
- UDP 통신 (라즈베리파이 → PC: 영상, PC → 라즈베리파이: 모터 명령)
- R키로 녹화 토글, WASD (동시 입력 지원)로 조종
- 저장 경로: `C:/PROJECT7/drive_dataset/YYYYMMDD_HHMMSS/`
- 저장 형식: `frame_000001.jpg` + `frame_000001.txt` (모터값, 키 조합)
- **stop 프레임은 저장 안 함** → 클래스 불균형 방지

### STEP 2 - 모방 학습 모델 학습 [완료 ✅]
- 학습 스크립트: `C:/PROJECT7/drive_train.py`
- 모델: MobileNetV3-Small (ImageNet pretrained → fine-tune)
- 입력: 224×224 RGB / 출력: 키 조합 분류 (w, a, d, wa, wd 등)
- 클래스: `['w', 's', 'a', 'd', 'wa', 'wd', 'sa', 'sd']` (데이터 없는 클래스 자동 제외)
- 클래스 불균형 → 가중치 자동 계산 (len(samples) / (N_cls × count))
- 데이터: `C:/PROJECT7/drive_dataset/`
- 저장: `C:/PROJECT7/drive_best.pth` (현재 자율주행 코드에서 사용 중)

### STEP 3 - 차선 세그멘테이션 학습 [완료 ✅]
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

### STEP 4 - 장애물 감지 모델 학습 [완료 ✅]
- 학습 스크립트: `C:/PROJECT7/object_train.py`
- 모델: YOLOv8n Detection
- 데이터: `C:/PROJECT7/object_dataset/20260406_181559/` (LabelMe rectangle JSON)
  - 포맷: LabelMe JSON, shape_type="rectangle"
  - 클래스: `"object"` / `"obstacle"` → ID 0 (단일 클래스)
  - 이미지 크기: 640×480
  - JSON이 없는 jpg는 자동 제외
- 학습 결과: `C:/PROJECT7/runs/object_det_v1/weights/best.pt`
- 자동 복사: `C:/PROJECT7/abcde.pt`

  **추가 촬영분 데이터 합산 방법:**
  ```python
  DATA_DIRS = [
      ROOT_DIR / "object_dataset" / "20260406_181559",
      ROOT_DIR / "object_dataset" / "새폴더명",   # 추가
  ]
  ```

### STEP 5 - 통합 자율주행 테스트 [완료 ✅]
- **v1**: `test_auto_drive(lane).py`  → 모방학습 + 차선 안전장치
- **v2**: `test_auto_drive(lane)2.py` → v1 + 횡단보도 정지
- **v3**: `test_auto_drive(lane)3.py` → v2 + 장애물 회피 (현재 최신)
- 키 조작: SPACE=일시정지/재개, L=차선안전장치토글, ESC=종료

---

## 주요 파일 구조
```
C:/PROJECT7/
├── CLAUDE.md                          ← 이 파일
├── README.MD                          ← 프로젝트 설명 (발표용)
│
├── [데이터 수집]
├── Pc_server(wasd).py                 ← STEP 1: 데이터 수집 서버 (PC)
├── Pi_drive(wasd).py                  ← STEP 1: 라즈베리파이 클라이언트
│
├── [학습]
├── drive_train.py                     ← STEP 2: 모방 학습 훈련 (PC)
├── lane_train.py                      ← STEP 3: 차선 세그멘테이션 학습 (PC)
├── object_train.py                    ← STEP 4: 장애물 감지 모델 학습 (PC)
│
├── [테스트]
├── test_auto_drive(wasd).py           ← 모방학습 단독 테스트
├── test_auto_drive(lane).py           ← v1: 모방학습 + 차선
├── test_auto_drive(lane)2.py          ← v2: v1 + 횡단보도
├── test_auto_drive(lane)3.py          ← v3: 최종 통합 (현재 사용)
│
├── [학습 결과 모델]
├── drive_best.pth                     ← 모방학습 모델 (drive_train.py 저장, 현재 사용)
├── abcde.pt                           ← 장애물 감지 모델 (object_train.py 자동 복사)
│
├── [데이터셋]
├── drive_dataset/                    ← 모방학습 수집 데이터 (세션별 하위폴더)
├── lane_dataset/                      ← 차선 학습 데이터 (LabelMe JSON)
│   ├── train/images/ & labels/
│   └── valid/images/ & labels/
├── object_dataset/                    ← 장애물 감지 학습 원본 데이터 (라벨링 원본, 필수 보관)
│   └── 20260406_181559/               ← jpg + LabelMe JSON 혼재
├── object_train_dataset/              ← object_train.py 실행 시 자동 생성 (삭제 후 재실행으로 복원 가능)
│   ├── train/images/ & labels/
│   └── valid/images/ & labels/
│
└── runs/
    ├── lane_seg_v1/weights/best.pt    ← 차선 모델 v1 (Mask mAP50=0.932)
    ├── lane_seg_v3/weights/best.pt    ← 차선 모델 v3 (crosswalk 추가, 현재 사용)
    └── object_det_v1/weights/best.pt  ← 장애물 모델
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

WASD 키 → 모터 매핑 (test_auto_drive(lane)3.py 기준):
```
W      → (-90,  -90)   전진
S      → ( 90,   90)   후진
A      → ( 30, -120)   좌회전 (제자리)
D      → (-120,  30)   우회전 (제자리)
W + A  → (-100,  -30)  전진+좌
W + D  → (-30,  -100)  전진+우
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

### 장애물 감지 설정
```python
OBJECT_CONF_THR       = 0.60   # 감지 최소 confidence
OBJECT_MIN_BOX_AREA   = 2000   # 최소 bbox 면적(px²) - 먼 물체 무시
OBJECT_CONFIRM_FRAMES = 1      # N프레임 연속 감지 시 회피 트리거
```

### 장애물 회피 5단계 순서 및 시간
```python
OBJECT_STOP_SEC      = 0.0    # ① 브레이킹 (0=즉시 조향으로 전환)
OBJECT_STEER_SEC     = 1.3    # ② 회피 방향으로 조향
OBJECT_FORWARD_SEC   = 1.0    # ③ 직진 (장애물 측면 통과)
OBJECT_RETURN_SEC_A  = 1.3    # ④-a 복귀 조향: 'a'(좌) 방향
OBJECT_RETURN_SEC_D  = 1.8    # ④-d 복귀 조향: 'd'(우) 방향
OBJECT_POST_FWD_SEC  = 0.8    # ⑤ 복귀 조향 후 직진
OBJECT_COOLDOWN_SEC  = 4.0    # 복귀 완료 후 재감지 방지 쿨다운
OBJECT_AVOID_SCALE_D = 0.6    # 'd'(우회피) 조향 강도
OBJECT_AVOID_SCALE_A = 0.6    # 'a'(좌회피) 조향 강도
```

**회피 방향 결정:** bbox 중심 x < 화면 중앙 → 왼쪽 장애물 → 오른쪽('d')으로 회피

**한계:** 카메라만으로는 멀리서 감지·회피가 어려움 → 초음파 센서 병행 필요

### 횡단보도 설정
```python
CROSSWALK_STOP_SEC     = 3.0   # 정지 시간
CROSSWALK_COOLDOWN_SEC = 30.0  # 재감지 방지 쿨다운
CROSSWALK_CONF_THR     = 0.45  # 감지 최소 confidence
```

### 차선 안전장치 설정
```python
LANE_THRESHOLD      = 130   # 이탈 판단 기준 (px, 중앙에서의 거리)
MIN_MASK_PIX        = 300   # 이 픽셀 수 미만이면 차선 미검출 처리
DEVIATION_COUNT     = 3     # N프레임 연속 이탈 시 보정 시작
CORRECTION_COOLDOWN = 1.0   # 보정 후 재보정까지 최소 간격 (초)
```

---

## lane_train.py 핵심 설계 결정
- **LabelMe JSON → YOLO seg 자동 변환**: `convert_json_to_yolo()` 함수
- **train/valid 자동 분리**: `split_dataset()` (valid ratio=0.2)
- **data.yaml 절대경로 사용** (상대경로 쓰면 오류)
- **optimizer=AdamW**, LR0=0.001, BATCH=16, workers=0 (Windows 필수)
- **mosaic=1.0**: 소규모 데이터 과적합 방지

## drive_train.py 핵심 설계 결정
- **DATA_ROOT**: `C:/PROJECT7/drive_dataset` (코드 내 경로 확인 후 실행)
- **수평 flip 금지**: 좌/우 라벨이 달라서 flip augmentation 쓰면 안 됨
- **밝기/대비 augmentation만 적용** (ColorJitter)
- **클래스 가중치 자동 계산**: `len(samples) / (NUM_CLASSES × counts[c])`
- **val_ratio=0.15**: 전체의 15%를 검증셋으로 사용
- **CosineAnnealingLR**: 학습률 스케줄러

## object_train.py 핵심 설계 결정
- **LabelMe rectangle JSON → YOLO detection txt 자동 변환**
  - 두 포인트(좌상단/우하단) → `cx cy w h` 정규화
- **task="detect"** (seg 아님) → 자율주행 코드가 boxes만 사용
- **CLASS_MAP**: `"object"` → 0, `"obstacle"` → 0
- **best.pt → abcde.pt 자동 복사** (자율주행 코드 경로 맞춤)

---

## 차선 detection이 잘 안 될 때 - fine-tuning 방법
```
기존 best.pt: C:/PROJECT7/runs/lane_seg_v3/weights/best.pt
새 결과 이름: lane_seg_v4
LR=0.0001, epoch=50
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
| OMP 중복 오류 | Windows + conda 환경 충돌 | `KMP_DUPLICATE_LIB_OK=TRUE` 추가 |
| 장애물 오감지 빈번 | 단일 프레임 노이즈 | OBJECT_CONFIRM_FRAMES로 연속 감지 필터링 |
| 장애물 회피 후 차선 복귀 어려움 | 단순 조향만으로는 부족 | 5단계 기동 (조향→직진→복귀조향→직진) |
| 먼 물체도 감지 | bbox 면적 필터 없음 | OBJECT_MIN_BOX_AREA=2000 추가 |
