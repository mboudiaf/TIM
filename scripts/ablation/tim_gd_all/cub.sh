META_TEST_ITER="1000"

# Only XEnt
python3 -m src.main \
        -F logs/ablation/tim_gd_all/cub \
        with dataset.path="data/cub/CUB_200_2011/images" \
        ckpt_path="checkpoints/cub/softmax/resnet18" \
        dataset.split_dir="split/cub" \
        model.arch='resnet18' \
        model.num_classes=100 \
        evaluate=True \
        eval.method='tim_gd' \
        tim.finetune_encoder=True \
        tim.iter=25 \
        tim.lr=5e-5 \
        eval.meta_test_iter=$META_TEST_ITER \
        tim.loss_weights="[0.1, 0., 0.]" \


# No H(Y)
python3 -m src.main \
        -F logs/ablation/tim_gd_all/cub \
        with dataset.path="data/cub/CUB_200_2011/images" \
        ckpt_path="checkpoints/cub/softmax/resnet18" \
        dataset.split_dir="split/cub" \
        model.arch='resnet18' \
        model.num_classes=100 \
        evaluate=True \
        eval.method='tim_gd' \
        tim.finetune_encoder=True \
        tim.iter=25 \
        tim.lr=5e-5 \
        eval.meta_test_iter=$META_TEST_ITER \
        tim.loss_weights="[0.1, 0., 0.1]" \

# No H(Y|X)
python3 -m src.main \
        -F logs/ablation/tim_gd_all/cub \
        with dataset.path="data/cub/CUB_200_2011/images" \
        ckpt_path="checkpoints/cub/softmax/resnet18" \
        dataset.split_dir="split/cub" \
        model.arch='resnet18' \
        model.num_classes=100 \
        evaluate=True \
        eval.method='tim_gd' \
        tim.finetune_encoder=True \
        tim.iter=25 \
        tim.lr=5e-5 \
        eval.meta_test_iter=$META_TEST_ITER \
        tim.loss_weights="[0.1, 1.0, 0.]" \

# No H(Y|X)
python3 -m src.main \
        -F logs/ablation/tim_gd_all/cub \
        with dataset.path="data/cub/CUB_200_2011/images" \
        ckpt_path="checkpoints/cub/softmax/resnet18" \
        dataset.split_dir="split/cub" \
        model.arch='resnet18' \
        model.num_classes=100 \
        evaluate=True \
        eval.method='tim_gd' \
        tim.finetune_encoder=True \
        tim.iter=25 \
        tim.lr=5e-5 \
        eval.meta_test_iter=$META_TEST_ITER \
        tim.loss_weights="[0.1, 1.0, 0.1]" \