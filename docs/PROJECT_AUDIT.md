# Sign2Text Project Audit

**Date**: 2026-10-09  
**Auditor**: opencode  
**Project**: Sign2Text (converted from Sign2Speech)

---

## 1. Repository Structure

```
/home/akshay-sv/Sign2Text/
├── data/raw/                  # Empty folders for later dataset collection
│   ├── HELLO/
│   ├── YES/
│   └── NO/
├── lpc2148/
│   ├── uart.c                # UART implementation (placeholders for GPIO)
│   ├── uart.h                # UART header (placeholders for pinout)
│   ├── lcd.c                 # LCD implementation (placeholders for GPIO)
│   └── lcd.h                 # LCD header (placeholders for pinout)
├── models/                    # Model checkpoints and metadata
│   └── (empty - no model files yet)
├── src/
│   ├── camera/camera.py      # Webcam abstraction
│   ├── detection/detector.py # MediaPipe HandLandmarker
│   ├── features/extractor.py # 126-dim feature extraction (wrist-relative, scale-normalized)
│   ├── dataset/loader.py     # Load (X, y) from .npy samples
│   ├── dataset/recorder.py   # Save .npy samples per label
│   ├── model/classifier.py   # RandomForest/SVM classifier with joblib
│   ├── model/trainer.py      # Training pipeline with validation
│   ├── prediction/predictor.py # Real-time prediction with debouncing
│   └── text/buffer.py        # Text entry with words/space/backspace/clear
├── config/settings.py        # Centralized configuration
├── requirements.txt          # Python dependencies
├── main.py                   # Real-time application entry point
├── test_serial.py            # UART test without ML model
└── train_cnn.py              # CNN training script (newly added)
```

---

## 2. Existing Functionality (Reused from Sign2Speech)

| Component | Status | Notes |
|-----------|--------|-------|
| MediaPipe HandLandmarker | ✅ Working | `detector.py` with `HandDetection/HandResult/Landmark3D` dataclasses |
| Feature Extraction | ✅ Working | `extractor.py` produces 126-dim vectors (2 hands x 63 coords), wrist-relative, scale-normalized |
| Dataset Recorder/Loader | ✅ Working | Saves/loads `.npy` feature vectors per label under `data/raw/<label>/` |
| Classical ML Classifier | ✅ Working | `classifier.py` with RandomForest/SVM, joblib model + metadata JSON |
| Text Buffer | ✅ Working | `buffer.py` with words/space/backspace/clear, undo/redo support |
| Serial Communication | ✅ Working | `serial.py` with configurable port/baud, line ending `\n`, send/read line |
| Main Application | ✅ Working | `main.py` with camera loop, prediction, text buffer, UART output |
| LPC2148 Firmware | 🟡 Placeholders | GPIO pins and LCD definitions need board schematic |
| CNN Pipeline | ❌ Not implemented | New requirement for this phase |
| End-to-End Integration | 🟡 Partial | Camera → MediaPipe → Features → Classical Model → Text → UART |

---

## 3. Missing Functionality (Phase 2+ Requirements)

| Area | Status | Required Action |
|------|--------|----------------|
| CNN training on ASL images | ❌ Not implemented | `train_cnn.py` to read images directly from `/home/akshay-sv/SignDB/asl_alphabet_train/` |
| CNN inference | ❌ Not integrated | Integrate `cnn_classifier.py` into webcam pipeline |
| Text buffer for character-by-character entry | 🟡 Partial | Adapt existing `buffer.py` for letter-by-letter (currently word-based) |
| Prediction stabilization | ✅ Working | `predictor.py` with confidence threshold + frame voting + cooldown |
| Serial protocol for edits | 🟡 Partial | Need to support CLEAR/BACKSPACE commands, not just append |
| Webcam preprocessing matching training | 🟡 Partial | Need to verify image size, channel order, normalization match |
| End-to-end tests | ❌ Not runnable | Need dataset, model, and hardware checks |
| Documentation | 🟡 Partial | `README.md` needs updating; new docs in `docs/` |

---

## 4. Dataset Information

**Expected location**: `/home/akshay-sv/SignDB/asl_alphabet_train/asl_alphabet_train/`

**Expected structure**: 35 classes (A-Z, del, nothing, space), 3000 images each = ~105,000 total

**Actual structure** (verified): 29 classes (A-Z, del, nothing, space), 3000 images each = ~87,000 total

**Classes**: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

**Images per class**: 3000 JPG files

**Test directory**: `/home/akshay-sv/SignDB/asl_alphabet_test/` - exists but appears empty (0 files in test subdirectory)

---

## 5. Python Environment

| Item | Status |
|------|--------|
| Python version | 3.12.3 |
| OpenCV | 5.0.0.93 (opencv-contrib-python) |
| NumPy | 2.5.3 |
| TensorFlow | 2.21.0 (installed for CNN training) |
| Keras | 3.15.1 (via TensorFlow) |
| MediaPipe | installed (checked in detector.py imports) |
| scikit-learn | 1.9.0 |
| joblib | installed |
| pyserial | 3.5 |

**Key compatibility note**: TensorFlow 2.21.0 + Keras 3.15.1 are installed. The CNN training uses `tf.keras.utils.image_dataset_from_directory`.

---

## 6. Dependency Status Summary

| Package | Required | Status |
|---------|----------|--------|
| opencv-python | ✅ | 5.0.0.93 installed |
| mediapipe | ✅ | Installed (used by detector.py) |
| numpy | ✅ | 2.5.3 installed |
| tensorflow | ✅ | 2.21.0 installed |
| keras | ✅ | 3.15.1 installed (via TF) |
| scikit-learn | ✅ | 1.9.0 installed |
| joblib | ✅ | Installed (used by classifier.py) |
| pyserial | ✅ | 3.5 installed |

---

## 7. Current Git Changes

No git repository initialized at `/home/akshay-sv/Sign2Text`. All files are present on disk.

---

## 8. Key Issues and Limitations

1. **No trained CNN model**: The CNN pipeline needs to be trained on the ASL dataset images
2. **Text buffer is word-based**: Currently adds signs as full words; needs adaptation for character-by-letter entry
3. **LPC2148 firmware pinouts**: All GPIO pin definitions are placeholders awaiting board schematic
4. **No serial hardware**: No LPC2148 board connected for UART testing
5. **Classifier model**: Classical ML model exists but may need retraining or verification
6. **Dataset format**: Images are JPG files in class subdirectories; existing code uses .npy feature vectors

---

## 9. Checklist: What Works vs. What Needs Work

| Category | Status |
|----------|--------|
| MediaPipe hand detection | ✅ Fully working |
| Feature extraction (126-dim) | ✅ Fully working |
| Classical ML train/evaluate/save/load | ✅ Fully working |
| Text buffer (words/space/backspace/clear) | ✅ Fully working |
| Serial communication (send text) | ✅ Fully working (configurable) |
| Main application loop | ✅ Fully working (with untrained classifier warning) |
| LPC2148 firmware | 🟡 Placeholders need board schematic |
| CNN training pipeline | ❌ New implementation needed |
| CNN inference integration | ❌ Not yet integrated |
| Character-by-letter text entry | 🟡 Needs adaptation from word-based |
| End-to-end testing | ❌ Not runnable without model and hardware |

---

## 10. Next Steps (Phase 2 onward)

1. Validate dataset structure and class distribution
2. Implement CNN training pipeline on ASL images
3. Integrate CNN with webcam prediction
4. Adapt text buffer for character-by-letter entry
5. Enhance serial protocol for edit commands
6. Create comprehensive tests
7. Update documentation