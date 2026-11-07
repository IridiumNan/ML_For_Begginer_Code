#!/usr/bin/env bash

# for strategy invsqrt
# --alpha parameters will be a0, scale, floor

python train.py --strategy invsqrt --alpha 0.8283 367.9 0.001 \
    --gamma 0.865 --epsilon 0.176 --punish -28.5 \
    --discretizer coarse --episodes 100000 --eval-episodes 100
