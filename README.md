# SSISv2 + SAM2 Video Shadow-Object Detection Pipeline

動画中の物体とその影をペアで検出・追跡するパイプライン（方法3）。

- **SSISv2**: キーフレームで物体-影ペアを検出
- **SAM2 Video Predictor**: 検出されたマスクを全フレームに伝播

## セットアップ

### 1. リポジトリのクローンとモデルダウンロード

```bash
bash setup.sh
```

### 2. SSISv2 のインストール

```bash
cd SSIS
pip install -r requirement.txt
python setup.py build develop
cd ..
```

> **注意:** CUDA toolkit と C++ コンパイラが必要。Windows では WSL2 推奨。

### 3. SAM2 のインストール

```bash
cd sam2
pip install -e .
cd ..
```

> **要件:** Python >= 3.10, PyTorch >= 2.5.1

### 4. SSISv2 の重みファイル (手動ダウンロード)

以下の Google Drive から `model_ssisv2_final.pth` をダウンロード:

https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP

配置先:
```
SSIS/tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/model_ssisv2_final.pth
```

## 使い方

### 基本実行

```bash
python pipeline.py --video input.mp4 --output ./output
```

### オプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--video` | (必須) | 入力動画パス |
| `--output` | `./output` | 出力ディレクトリ |
| `--keyframe` | `0` | SSISv2 を実行するフレームのインデックス |
| `--threshold` | `0.3` | SSISv2 の検出信頼度閾値 |
| `--sam2-model` | `large` | SAM2 モデルサイズ (`tiny`/`small`/`base_plus`/`large`) |
| `--device` | `cuda` | デバイス |
| `--no-vis` | `false` | 可視化画像の保存をスキップ |

### 出力構造

```
output/
├── frames/           # 抽出されたフレーム (JPEG)
├── masks/            # 各フレーム×各物体のマスク (PNG)
│   ├── frame00000_obj0.png   # Pair 0 の影マスク
│   ├── frame00000_obj1.png   # Pair 0 の物体マスク
│   ├── frame00001_obj0.png
│   └── ...
├── visualization/    # マスクオーバーレイ画像
└── pairs.json        # ペア対応メタデータ
```

### pairs.json の形式

```json
{
  "pairs": [
    {
      "shadow_obj_id": 0,
      "object_obj_id": 1,
      "shadow_score": 0.85,
      "object_score": 0.92
    }
  ],
  "total_frames": 120
}
```

## 個別実行

### SSISv2 のみ（単一画像）

```bash
python ssis_inference.py --image photo.jpg --output ./output_ssis
```

### パイプラインの仕組み

```
入力動画
  │
  ├─[フレーム抽出]──→ frames/ (JPEG)
  │
  ├─[SSISv2]─────→ キーフレームで影-物体ペア検出
  │                  出力: shadow_mask + object_mask (per pair)
  │
  ├─[SAM2 Video]──→ 全マスクを全フレームに伝播
  │                  入力: 初期マスク (keyframe)
  │                  出力: 全フレームのマスク
  │
  └─[保存]────────→ masks/ + pairs.json + visualization/
```
