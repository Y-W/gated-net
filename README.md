# gated-net 
## Adapted from <https://github.com/tensorflow/models/tree/master/slim>


Version Details:

train_v*.py, eval_v*.py

Hard gating: Straight-through estimator + boxcar (-1 ~ 1)

| Name  | Resnet Arch | Gating | Bias  | Fully trainable |
| :---: | :---------: | :----: | :---: | :-------------: |
| v1    | resnetV1    | soft   | yes   | no              |
| v2    | resnetV1    | hard   | yes   | no              |
| v3    | resnetV1    | soft   | yes   | yes             |
| v4    | resnetV1    | hard   | yes   | yes             |

