#!/bin/bash
# Mode-specific 훈련 및 평가 스크립트

# =============================================================================
# OVSS Mode (Open-Vocabulary Semantic Segmentation)
# =============================================================================

echo "🔧 OVSS Mode: Open-Vocabulary Semantic Segmentation"
echo "  Known: 0~74 vs Unknown: 75~149 (총 150 클래스)"

# 1. OVSS 훈련
train_ovss() {
    echo "🚀 Training OVSS mode..."
    python ow_train_net.py \
        --config configs/ow_vitb_384.yaml \
        --num-gpus 1 \
        --dist-url "auto" \
        OUTPUT_DIR train_ovss_$(date +%m%d%H) \
        MODEL.SEM_SEG_HEAD.EVALUATION_MODE "OVSS"
}

# 2. OVSS 평가
eval_ovss() {
    MODEL_PATH=$1
    echo "📊 Evaluating OVSS mode with model: $MODEL_PATH"

    python ow_train_net.py \
        --config configs/ow_vitb_384.yaml \
        --num-gpus 1 \
        --dist-url "auto" \
        --eval-only \
        OUTPUT_DIR eval_ovss_$(date +%m%d%H) \
        MODEL.WEIGHTS $MODEL_PATH \
        MODEL.SEM_SEG_HEAD.EVALUATION_MODE "OVSS" \
        TEST.SLIDING_WINDOW "True" \
        MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]"
}

# =============================================================================
# OWSS Mode (Open-World Semantic Segmentation)
# =============================================================================

echo "🔧 OWSS Mode: Open-World Semantic Segmentation"
echo "  Known: 0~74 vs Unknown: 150 (리맵: 75~149 → 150)"

# 3. OWSS 훈련
train_owss() {
    echo "🚀 Training OWSS mode..."
    python ow_train_net.py \
        --config configs/ow_vitb_384.yaml \
        --num-gpus 1 \
        --dist-url "auto" \
        OUTPUT_DIR train_owss_$(date +%m%d%H) \
        MODEL.SEM_SEG_HEAD.EVALUATION_MODE "OWSS"
}

# 4. OWSS 평가
eval_owss() {
    MODEL_PATH=$1
    echo "📊 Evaluating OWSS mode with model: $MODEL_PATH"

    python ow_train_net.py \
        --config configs/ow_vitb_384.yaml \
        --num-gpus 1 \
        --dist-url "auto" \
        --eval-only \
        OUTPUT_DIR eval_owss_$(date +%m%d%H) \
        MODEL.WEIGHTS $MODEL_PATH \
        MODEL.SEM_SEG_HEAD.EVALUATION_MODE "OWSS" \
        TEST.SLIDING_WINDOW "True" \
        MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]"
}

# =============================================================================
# 비교 평가
# =============================================================================

# 5. 두 모드 성능 비교
compare_modes() {
    MODEL_PATH=$1
    echo "🔍 Comparing OVSS vs OWSS modes with model: $MODEL_PATH"

    echo "1. OVSS 평가 중..."
    eval_ovss $MODEL_PATH > ovss_results.log 2>&1

    echo "2. OWSS 평가 중..."
    eval_owss $MODEL_PATH > owss_results.log 2>&1

    echo "3. 결과 비교:"
    echo "OVSS Results:"
    grep "copypaste:" ovss_results.log | tail -1
    echo "OWSS Results:"
    grep "copypaste:" owss_results.log | tail -1
}

# =============================================================================
# 메인 실행 함수
# =============================================================================

main() {
    case $1 in
        "train_ovss")
            train_ovss
            ;;
        "train_owss")
            train_owss
            ;;
        "eval_ovss")
            if [ -z "$2" ]; then
                echo "Usage: $0 eval_ovss <model_path>"
                exit 1
            fi
            eval_ovss $2
            ;;
        "eval_owss")
            if [ -z "$2" ]; then
                echo "Usage: $0 eval_owss <model_path>"
                exit 1
            fi
            eval_owss $2
            ;;
        "compare")
            if [ -z "$2" ]; then
                echo "Usage: $0 compare <model_path>"
                exit 1
            fi
            compare_modes $2
            ;;
        *)
            echo "Usage: $0 {train_ovss|train_owss|eval_ovss|eval_owss|compare} [model_path]"
            echo ""
            echo "Examples:"
            echo "  $0 train_ovss                    # OVSS 모드 훈련"
            echo "  $0 train_owss                    # OWSS 모드 훈련"
            echo "  $0 eval_ovss model_final.pth     # OVSS 모드 평가"
            echo "  $0 eval_owss model_final.pth     # OWSS 모드 평가"
            echo "  $0 compare model_final.pth       # 두 모드 성능 비교"
            echo ""
            echo "🔧 OVSS: 0~74 (known) vs 75~149 (unknown) - 총 150 클래스"
            echo "🔧 OWSS: 0~74 (known) vs 150 (unknown, 75~149 리맵) - 총 76 클래스"
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@"