python3 run_winclip.py \
    --datasets visa \
    --use-adapter true \
    --adapter-root-dir ./result_adapter \
    --shared-adapter true \
    --batch-size 32 \
    --gpu-id 0 \
    --use-cpu 0 \
    --pretrained_dataset /root/winclipbaseline/checkpoints/vit_b_16_plus_240-laion400m_e32-699c4b84.pt
