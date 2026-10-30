# Overnight results (fresh validation pass; runtime us / energy mJ per call)

## exp3_w2_linear_gelu_cublas 4096x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_last_pair_ordinary | 6370.049 | 1869.439 |  |
| compiled_full_ordinary | 6371.187 | 1872.6322 |  |
| compiled_first_pair_ordinary | 6387.651 | 1922.8435 |  |
| compiled_separate_ordinary | 6595.531 | 1966.6386 |  |
| eager_baseline | 6601.816 | 1976.2707 |  |
| compiled_first_pair_replay | 6842.395 | 1977.6922 |  |
| compiled_full_replay | 6837.915 | 1977.6964 |  |
| compiled_last_pair_replay | 6828.831 | 2023.6 |  |
| compiled_separate_replay | 7053.141 | 2075.0657 |  |

## exp4_w2_linear_gelu_triton_gemm 4096x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| eager_baseline | 6597.849 | 1986.7364 |  |
| compiled_last_pair_ordinary | 7060.987 | 2090.3879 |  |
| compiled_full_ordinary | 6909.521 | 2092.6225 |  |
| compiled_first_pair_ordinary | 7047.154 | 2131.0356 |  |
| compiled_first_pair_replay | 7431.295 | 2191.5791 |  |
| compiled_separate_ordinary | 7256.152 | 2196.5513 |  |
| compiled_last_pair_replay | 7459.978 | 2197.627 |  |
| compiled_full_replay | 7287.534 | 2207.6874 |  |
| compiled_separate_replay | 7644.349 | 2315.4582 |  |

## exp5_w3_layernorm_reduction 4096x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_full_ordinary | 392.629 | 93.5871 | yes |
| compiled_last_pair_ordinary | 603.375 | 135.7088 |  |
| compiled_first_pair_ordinary | 627.309 | 140.9423 |  |
| compiled_separate_ordinary | 840.842 | 187.4796 |  |
| compiled_full_replay | 891.299 | 201.3023 |  |
| eager_baseline | 956.636 | 213.6141 |  |
| compiled_last_pair_replay | 1085.695 | 236.0023 |  |
| compiled_first_pair_replay | 1125.24 | 249.5752 |  |
| compiled_separate_replay | 1321.342 | 290.5312 |  |

## exp6_w4_fan_out 4096x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_full_ordinary | 251.707 | 54.4303 | yes |
| compiled_first_pair_ordinary | 476.689 | 102.3527 |  |
| compiled_last_pair_ordinary | 486.184 | 103.4861 |  |
| compiled_full_replay | 497.219 | 103.9723 |  |
| compiled_separate_ordinary | 585.115 | 126.6844 |  |
| eager_baseline | 611.327 | 139.0186 |  |
| compiled_first_pair_replay | 721.141 | 154.4305 |  |
| compiled_last_pair_replay | 730.48 | 155.2214 |  |
| compiled_separate_replay | 828.783 | 178.6752 |  |

## exp7_w1_launch_bound 256x1024 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_full_replay | 37.275 | 3.1142 | yes |
| compiled_first_pair_replay | 37.308 | 3.2795 |  |
| compiled_last_pair_replay | 37.481 | 3.2945 |  |
| compiled_separate_replay | 37.731 | 3.5826 |  |
| eager_baseline | 64.874 | 5.8758 |  |
| compiled_full_ordinary | 114.099 | 8.9679 |  |
| compiled_first_pair_ordinary | 193.087 | 15.2213 |  |
| compiled_last_pair_ordinary | 193.288 | 15.458 |  |
| compiled_separate_ordinary | 269.179 | 21.3523 |  |

## exp8_w3_layernorm_small 32x1024 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_first_pair_replay | 34.895 | 2.791 |  |
| compiled_full_replay | 34.905 | 2.7953 |  |
| compiled_separate_replay | 35.492 | 2.8272 |  |
| compiled_last_pair_replay | 35.181 | 2.8435 |  |
| eager_baseline | 78.191 | 6.0408 |  |
| compiled_full_ordinary | 114.108 | 8.4797 |  |
| compiled_last_pair_ordinary | 194.045 | 14.8551 |  |
| compiled_first_pair_ordinary | 193.499 | 15.0218 |  |
| compiled_separate_ordinary | 268.381 | 20.6955 |  |

## exp9_w4_fan_out_launch_bound 256x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_full_replay | 33.697 | 4.7631 | yes |
| compiled_last_pair_replay | 33.807 | 6.1511 |  |
| compiled_first_pair_replay | 34.066 | 6.4302 |  |
| compiled_separate_replay | 34.419 | 6.5764 |  |
| eager_baseline | 64.26 | 10.2741 |  |
| compiled_full_ordinary | 105.375 | 10.7592 |  |
| compiled_last_pair_ordinary | 182.767 | 18.3643 |  |
| compiled_first_pair_ordinary | 194.005 | 19.3618 |  |
| compiled_separate_ordinary | 263.914 | 25.2911 |  |

## exp10_w5_softmax_small 32x1024 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_full_replay | 33.349 | 2.6805 | yes |
| compiled_last_pair_replay | 33.979 | 2.738 |  |
| compiled_first_pair_replay | 33.715 | 2.7583 |  |
| compiled_separate_replay | 34.038 | 2.7974 |  |
| eager_baseline | 61.202 | 4.997 |  |
| compiled_full_ordinary | 107.869 | 8.3404 |  |
| compiled_first_pair_ordinary | 185.168 | 14.2727 |  |
| compiled_last_pair_ordinary | 186.397 | 14.3148 |  |
| compiled_separate_ordinary | 256.812 | 19.92 |  |

## exp11_w2_gemm_bound 128x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_last_pair_replay | 261.034 | 77.8978 |  |
| compiled_full_replay | 260.171 | 78.0792 |  |
| compiled_full_ordinary | 260.739 | 78.2294 |  |
| compiled_last_pair_ordinary | 261.767 | 78.6569 |  |
| compiled_separate_replay | 263.731 | 78.8869 |  |
| eager_baseline | 265.333 | 79.2166 |  |
| compiled_first_pair_ordinary | 309.009 | 92.197 |  |
| compiled_first_pair_replay | 308.945 | 92.2412 |  |
| compiled_separate_ordinary | 322.6 | 92.7871 |  |

## exp12_w2_fuse_epilogue_decode 128x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_last_pair_replay | 262.755 | 77.9799 | yes |
| compiled_last_pair_ordinary | 263.858 | 78.728 |  |
| compiled_separate_replay | 265.776 | 79.1566 |  |
| eager_baseline | 267.321 | 80.2842 |  |
| compiled_first_pair_replay | 298.024 | 88.6419 |  |
| compiled_first_pair_ordinary | 298.58 | 89.6826 |  |
| compiled_full_ordinary | 300.669 | 89.6838 |  |
| compiled_full_replay | 300.887 | 90.3273 |  |
| compiled_separate_ordinary | 323.034 | 91.9704 |  |

## exp12_w2_fuse_epilogue_decode 16x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_full_replay | 115.725 | 34.5339 |  |
| compiled_first_pair_replay | 118.226 | 34.7507 |  |
| compiled_full_ordinary | 118.642 | 34.9893 |  |
| compiled_last_pair_replay | 141.356 | 42.2859 |  |
| compiled_first_pair_ordinary | 202.267 | 42.3471 |  |
| compiled_separate_replay | 143.042 | 42.9658 |  |
| eager_baseline | 145.491 | 42.9963 |  |
| compiled_last_pair_ordinary | 239.363 | 59.7225 |  |
| compiled_separate_ordinary | 315.337 | 65.9576 |  |

## exp12_w2_fuse_epilogue_decode 64x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_last_pair_replay | 144.844 | 42.8981 |  |
| compiled_separate_replay | 146.651 | 44.2354 |  |
| eager_baseline | 149.575 | 44.9442 |  |
| compiled_full_replay | 175.918 | 52.0625 |  |
| compiled_full_ordinary | 176.318 | 52.8343 |  |
| compiled_first_pair_replay | 177.933 | 53.3421 |  |
| compiled_first_pair_ordinary | 197.906 | 59.0183 |  |
| compiled_last_pair_ordinary | 234.721 | 68.1708 |  |
| compiled_separate_ordinary | 310.031 | 78.7763 |  |

## exp13_w6_transformer_mlp_decode 64x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_add+ln_mm_add_gelu_replay | 154.075 | 45.4222 |  |
| compiled_add+ln_mm_add+gelu_replay | 152.035 | 45.666 |  |
| compiled_add_ln_mm_add+gelu_replay | 154.519 | 46.3661 |  |
| compiled_separate_replay | 156.294 | 46.837 |  |
| eager_baseline | 164.883 | 49.0058 |  |
| compiled_add_ln_mm+add+gelu_replay | 182.247 | 53.7758 |  |
| compiled_add_ln+mm+add+gelu_replay | 182.877 | 53.8689 |  |
| compiled_add_ln_mm+add_gelu_replay | 184.018 | 54.1421 |  |
| compiled_full_replay | 180.533 | 54.145 |  |
| compiled_add_ln+mm_add+gelu_replay | 184.156 | 54.1635 |  |
| compiled_add+ln_mm+add+gelu_replay | 180.816 | 54.2259 |  |
| compiled_full_ordinary | 181.367 | 54.2738 |  |
| compiled_add+ln+mm+add_gelu_replay | 181.475 | 54.4544 |  |
| compiled_add+ln+mm_add+gelu_replay | 181.961 | 54.6865 |  |
| compiled_add+ln_mm+add_gelu_replay | 182.299 | 54.7124 |  |
| compiled_add_ln+mm+add_gelu_replay | 183.283 | 54.7663 |  |
| compiled_add+ln+mm_add_gelu_replay | 183.977 | 54.9649 |  |
| compiled_add_ln+mm_add_gelu_replay | 186.058 | 55.5381 |  |
| compiled_add+ln_mm+add+gelu_ordinary | 206.47 | 60.3139 |  |
| compiled_add_ln+mm+add+gelu_ordinary | 227.854 | 67.4699 |  |
| compiled_add+ln+mm_add+gelu_ordinary | 229.881 | 67.6588 |  |
| compiled_add+ln+mm+add_gelu_ordinary | 228.856 | 68.1787 |  |
| compiled_add_ln_mm+add+gelu_ordinary | 283.021 | 78.3735 |  |
| compiled_add+ln_mm_add+gelu_ordinary | 323.306 | 78.979 |  |
| compiled_add+ln_mm+add_gelu_ordinary | 285.959 | 79.2922 |  |
| compiled_add_ln+mm+add_gelu_ordinary | 307.502 | 80.1759 |  |
| compiled_add_ln+mm_add+gelu_ordinary | 302.067 | 81.0826 |  |
| compiled_add+ln+mm_add_gelu_ordinary | 305.794 | 81.7358 |  |
| compiled_add_ln_mm+add_gelu_ordinary | 361.689 | 85.8185 |  |
| compiled_add+ln_mm_add_gelu_ordinary | 403.587 | 86.242 |  |
| compiled_add_ln_mm_add+gelu_ordinary | 403.799 | 86.499 |  |
| compiled_add_ln+mm_add_gelu_ordinary | 379.894 | 86.5849 |  |
| compiled_separate_ordinary | 477.584 | 92.7187 |  |

## exp14_w6_transformer_mlp_prefill 512x4096 (validation pass)

| candidate | runtime us | energy mJ | selected |
|---|---|---|---|
| compiled_add+ln_mm_add+gelu_replay | 954.523 | 286.4367 |  |
| compiled_add+ln_mm_add+gelu_ordinary | 958.893 | 287.4841 |  |
| compiled_add_ln_mm_add+gelu_replay | 978.475 | 289.1379 |  |
| compiled_add_ln_mm_add+gelu_ordinary | 980.097 | 293.9626 |  |
| compiled_add+ln_mm_add_gelu_replay | 981.331 | 295.261 |  |
| compiled_separate_replay | 1004.811 | 296.6566 |  |
| compiled_add+ln_mm_add_gelu_ordinary | 986.064 | 296.8731 |  |
| compiled_separate_ordinary | 1008.503 | 304.2528 |  |
| eager_baseline | 1017.96 | 306.9584 |  |
| compiled_add+ln_mm+add+gelu_replay | 1099.76 | 324.9544 |  |
| compiled_full_replay | 1100.074 | 325.0705 |  |
| compiled_add+ln+mm+add_gelu_ordinary | 1121.64 | 328.8956 |  |
| compiled_add+ln_mm+add+gelu_ordinary | 1101.285 | 330.1748 |  |
| compiled_add+ln+mm_add+gelu_replay | 1116.759 | 330.2167 |  |
| compiled_add_ln+mm+add+gelu_replay | 1122.041 | 330.4484 |  |
| compiled_add_ln_mm+add+gelu_ordinary | 1123.4 | 330.9877 |  |
| compiled_add_ln_mm+add+gelu_replay | 1122.55 | 331.0869 |  |
| compiled_full_ordinary | 1101.004 | 331.6693 |  |
| compiled_add+ln+mm+add_gelu_replay | 1119.003 | 334.7492 |  |
| compiled_add+ln_mm+add_gelu_ordinary | 1121.028 | 335.406 |  |
| compiled_add_ln+mm_add+gelu_ordinary | 1143.101 | 336.2146 |  |
| compiled_add_ln+mm+add_gelu_replay | 1141.455 | 336.2383 |  |
| compiled_add+ln+mm_add+gelu_ordinary | 1120.263 | 336.3019 |  |
| compiled_add_ln+mm+add+gelu_ordinary | 1123.402 | 336.3776 |  |
| compiled_add+ln_mm+add_gelu_replay | 1118.808 | 336.4002 |  |
| compiled_add_ln+mm+add_gelu_ordinary | 1144.519 | 337.2288 |  |
| compiled_add_ln+mm_add+gelu_replay | 1139.895 | 341.4898 |  |
| compiled_add_ln_mm+add_gelu_replay | 1140.944 | 341.655 |  |
| compiled_add+ln+mm_add_gelu_replay | 1143.655 | 342.2261 |  |
| compiled_add_ln_mm+add_gelu_ordinary | 1144.553 | 343.3944 |  |
| compiled_add_ln+mm_add_gelu_replay | 1166.64 | 343.5221 |  |
| compiled_add+ln+mm_add_gelu_ordinary | 1148.644 | 344.4574 |  |
| compiled_add_ln+mm_add_gelu_ordinary | 1170.344 | 350.3917 |  |

