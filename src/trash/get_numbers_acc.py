import re

text = """ 
Epoch: 0
Train epoch: 0	Loss: 1.966 | Acc: 29.41% (14706/50000)
Test epoch: 0	Loss: 1.583 | Acc: 41.71% (4171/10000)
Saving..

Epoch: 1
Train epoch: 1	Loss: 1.426 | Acc: 47.78% (23891/50000)
Test epoch: 1	Loss: 1.380 | Acc: 50.70% (5070/10000)
Saving..

Epoch: 2
Train epoch: 2	Loss: 1.131 | Acc: 59.37% (29687/50000)
Test epoch: 2	Loss: 1.118 | Acc: 59.71% (5971/10000)
Saving..

Epoch: 3
Train epoch: 3	Loss: 0.943 | Acc: 66.64% (33322/50000)
Test epoch: 3	Loss: 0.969 | Acc: 67.00% (6700/10000)
Saving..

Epoch: 4
Train epoch: 4	Loss: 0.808 | Acc: 71.50% (35748/50000)
Test epoch: 4	Loss: 1.060 | Acc: 65.78% (6578/10000)

Epoch: 5
Train epoch: 5	Loss: 0.694 | Acc: 75.76% (37882/50000)
Test epoch: 5	Loss: 0.778 | Acc: 74.09% (7409/10000)
Saving..

Epoch: 6
Train epoch: 6	Loss: 0.629 | Acc: 78.28% (39142/50000)
Test epoch: 6	Loss: 0.669 | Acc: 77.55% (7755/10000)
Saving..

Epoch: 7
Train epoch: 7	Loss: 0.580 | Acc: 80.09% (40047/50000)
Test epoch: 7	Loss: 0.608 | Acc: 79.01% (7901/10000)
Saving..

Epoch: 8
Train epoch: 8	Loss: 0.541 | Acc: 81.28% (40640/50000)
Test epoch: 8	Loss: 0.888 | Acc: 71.27% (7127/10000)

Epoch: 9
Train epoch: 9	Loss: 0.515 | Acc: 82.26% (41131/50000)
Test epoch: 9	Loss: 0.686 | Acc: 77.44% (7744/10000)

Epoch: 10
Train epoch: 10	Loss: 0.492 | Acc: 83.09% (41545/50000)
Test epoch: 10	Loss: 0.856 | Acc: 72.40% (7240/10000)

Epoch: 11
Train epoch: 11	Loss: 0.475 | Acc: 83.73% (41864/50000)
Test epoch: 11	Loss: 0.554 | Acc: 81.44% (8144/10000)
Saving..

Epoch: 12
Train epoch: 12	Loss: 0.463 | Acc: 84.22% (42108/50000)
Test epoch: 12	Loss: 0.780 | Acc: 74.58% (7458/10000)

Epoch: 13
Train epoch: 13	Loss: 0.451 | Acc: 84.75% (42373/50000)
Test epoch: 13	Loss: 0.611 | Acc: 80.32% (8032/10000)

Epoch: 14
Train epoch: 14	Loss: 0.438 | Acc: 85.00% (42499/50000)
Test epoch: 14	Loss: 0.561 | Acc: 81.43% (8143/10000)

Epoch: 15
Train epoch: 15	Loss: 0.433 | Acc: 85.39% (42695/50000)
Test epoch: 15	Loss: 0.787 | Acc: 74.89% (7489/10000)

Epoch: 16
Train epoch: 16	Loss: 0.419 | Acc: 85.62% (42808/50000)
Test epoch: 16	Loss: 0.492 | Acc: 83.45% (8345/10000)
Saving..

Epoch: 17
Train epoch: 17	Loss: 0.408 | Acc: 85.98% (42992/50000)
Test epoch: 17	Loss: 0.603 | Acc: 80.55% (8055/10000)

Epoch: 18
Train epoch: 18	Loss: 0.400 | Acc: 86.38% (43189/50000)
Test epoch: 18	Loss: 0.525 | Acc: 81.95% (8195/10000)

Epoch: 19
Train epoch: 19	Loss: 0.398 | Acc: 86.36% (43181/50000)
Test epoch: 19	Loss: 0.485 | Acc: 83.71% (8371/10000)
Saving..

Epoch: 20
Train epoch: 20	Loss: 0.392 | Acc: 86.60% (43298/50000)
Test epoch: 20	Loss: 0.560 | Acc: 81.69% (8169/10000)

Epoch: 21
Train epoch: 21	Loss: 0.381 | Acc: 86.90% (43448/50000)
Test epoch: 21	Loss: 0.611 | Acc: 79.93% (7993/10000)

Epoch: 22
Train epoch: 22	Loss: 0.378 | Acc: 87.09% (43544/50000)
Test epoch: 22	Loss: 0.487 | Acc: 84.01% (8401/10000)
Saving..

Epoch: 23
Train epoch: 23	Loss: 0.374 | Acc: 87.23% (43614/50000)
Test epoch: 23	Loss: 0.691 | Acc: 78.35% (7835/10000)

Epoch: 24
Train epoch: 24	Loss: 0.366 | Acc: 87.50% (43751/50000)
Test epoch: 24	Loss: 0.625 | Acc: 80.09% (8009/10000)

Epoch: 25
Train epoch: 25	Loss: 0.372 | Acc: 87.33% (43663/50000)
Test epoch: 25	Loss: 1.021 | Acc: 70.87% (7087/10000)

Epoch: 26
Train epoch: 26	Loss: 0.360 | Acc: 87.77% (43884/50000)
Test epoch: 26	Loss: 0.569 | Acc: 81.64% (8164/10000)

Epoch: 27
Train epoch: 27	Loss: 0.353 | Acc: 87.98% (43992/50000)
Test epoch: 27	Loss: 0.436 | Acc: 85.32% (8532/10000)
Saving..

Epoch: 28
Train epoch: 28	Loss: 0.356 | Acc: 87.88% (43942/50000)
Test epoch: 28	Loss: 0.531 | Acc: 82.48% (8248/10000)

Epoch: 29
Train epoch: 29	Loss: 0.347 | Acc: 88.16% (44078/50000)
Test epoch: 29	Loss: 0.520 | Acc: 83.07% (8307/10000)

Epoch: 30
Train epoch: 30	Loss: 0.347 | Acc: 88.20% (44101/50000)
Test epoch: 30	Loss: 0.468 | Acc: 84.17% (8417/10000)

Epoch: 31
Train epoch: 31	Loss: 0.345 | Acc: 88.24% (44121/50000)
Test epoch: 31	Loss: 0.458 | Acc: 84.74% (8474/10000)

Epoch: 32
Train epoch: 32	Loss: 0.341 | Acc: 88.47% (44235/50000)
Test epoch: 32	Loss: 0.492 | Acc: 83.94% (8394/10000)

Epoch: 33
Train epoch: 33	Loss: 0.340 | Acc: 88.45% (44227/50000)
Test epoch: 33	Loss: 0.481 | Acc: 84.45% (8445/10000)

Epoch: 34
Train epoch: 34	Loss: 0.337 | Acc: 88.50% (44251/50000)
Test epoch: 34	Loss: 0.516 | Acc: 83.14% (8314/10000)

Epoch: 35
Train epoch: 35	Loss: 0.334 | Acc: 88.64% (44318/50000)
Test epoch: 35	Loss: 0.900 | Acc: 73.28% (7328/10000)

Epoch: 36
Train epoch: 36	Loss: 0.333 | Acc: 88.51% (44255/50000)
Test epoch: 36	Loss: 0.456 | Acc: 84.88% (8488/10000)

Epoch: 37
Train epoch: 37	Loss: 0.329 | Acc: 88.72% (44362/50000)
Test epoch: 37	Loss: 0.496 | Acc: 83.70% (8370/10000)

Epoch: 38
Train epoch: 38	Loss: 0.331 | Acc: 88.61% (44306/50000)
Test epoch: 38	Loss: 0.464 | Acc: 84.37% (8437/10000)

Epoch: 39
Train epoch: 39	Loss: 0.319 | Acc: 89.14% (44571/50000)
Test epoch: 39	Loss: 0.469 | Acc: 83.87% (8387/10000)

Epoch: 40
Train epoch: 40	Loss: 0.321 | Acc: 88.97% (44483/50000)
Test epoch: 40	Loss: 0.811 | Acc: 76.48% (7648/10000)

Epoch: 41
Train epoch: 41	Loss: 0.318 | Acc: 89.20% (44600/50000)
Test epoch: 41	Loss: 0.463 | Acc: 84.42% (8442/10000)

Epoch: 42
Train epoch: 42	Loss: 0.318 | Acc: 89.22% (44609/50000)
Test epoch: 42	Loss: 0.446 | Acc: 85.19% (8519/10000)

Epoch: 43
Train epoch: 43	Loss: 0.312 | Acc: 89.24% (44620/50000)
Test epoch: 43	Loss: 0.452 | Acc: 85.12% (8512/10000)

Epoch: 44
Train epoch: 44	Loss: 0.314 | Acc: 89.31% (44656/50000)
Test epoch: 44	Loss: 0.486 | Acc: 84.50% (8450/10000)

Epoch: 45
Train epoch: 45	Loss: 0.315 | Acc: 89.26% (44630/50000)
Test epoch: 45	Loss: 0.434 | Acc: 85.27% (8527/10000)

Epoch: 46
Train epoch: 46	Loss: 0.304 | Acc: 89.61% (44805/50000)
Test epoch: 46	Loss: 0.604 | Acc: 81.99% (8199/10000)

Epoch: 47
Train epoch: 47	Loss: 0.308 | Acc: 89.50% (44750/50000)
Test epoch: 47	Loss: 0.470 | Acc: 84.99% (8499/10000)

Epoch: 48
Train epoch: 48	Loss: 0.301 | Acc: 89.60% (44802/50000)
Test epoch: 48	Loss: 0.397 | Acc: 86.93% (8693/10000)
Saving..

Epoch: 49
Train epoch: 49	Loss: 0.305 | Acc: 89.59% (44797/50000)
Test epoch: 49	Loss: 0.612 | Acc: 80.77% (8077/10000)

Epoch: 50
Train epoch: 50	Loss: 0.297 | Acc: 89.81% (44903/50000)
Test epoch: 50	Loss: 0.554 | Acc: 82.67% (8267/10000)

Epoch: 51
Train epoch: 51	Loss: 0.298 | Acc: 89.76% (44880/50000)
Test epoch: 51	Loss: 0.566 | Acc: 82.42% (8242/10000)

Epoch: 52
Train epoch: 52	Loss: 0.296 | Acc: 89.83% (44917/50000)
Test epoch: 52	Loss: 0.589 | Acc: 81.05% (8105/10000)

Epoch: 53
Train epoch: 53	Loss: 0.293 | Acc: 90.04% (45022/50000)
Test epoch: 53	Loss: 0.464 | Acc: 84.88% (8488/10000)

Epoch: 54
Train epoch: 54	Loss: 0.289 | Acc: 90.08% (45039/50000)
Test epoch: 54	Loss: 0.558 | Acc: 82.32% (8232/10000)

Epoch: 55
Train epoch: 55	Loss: 0.287 | Acc: 90.18% (45089/50000)
Test epoch: 55	Loss: 0.455 | Acc: 85.08% (8508/10000)

Epoch: 56
Train epoch: 56	Loss: 0.293 | Acc: 90.11% (45054/50000)
Test epoch: 56	Loss: 0.508 | Acc: 83.43% (8343/10000)

Epoch: 57
Train epoch: 57	Loss: 0.284 | Acc: 90.39% (45196/50000)
Test epoch: 57	Loss: 0.424 | Acc: 86.36% (8636/10000)

Epoch: 58
Train epoch: 58	Loss: 0.285 | Acc: 90.32% (45159/50000)
Test epoch: 58	Loss: 0.537 | Acc: 82.72% (8272/10000)

Epoch: 59
Train epoch: 59	Loss: 0.284 | Acc: 90.22% (45109/50000)
Test epoch: 59	Loss: 0.476 | Acc: 84.50% (8450/10000)

Epoch: 60
Train epoch: 60	Loss: 0.280 | Acc: 90.43% (45215/50000)
Test epoch: 60	Loss: 0.602 | Acc: 81.61% (8161/10000)

Epoch: 61
Train epoch: 61	Loss: 0.279 | Acc: 90.57% (45283/50000)
Test epoch: 61	Loss: 0.401 | Acc: 86.45% (8645/10000)

Epoch: 62
Train epoch: 62	Loss: 0.279 | Acc: 90.54% (45269/50000)
Test epoch: 62	Loss: 0.411 | Acc: 86.40% (8640/10000)

Epoch: 63
Train epoch: 63	Loss: 0.278 | Acc: 90.51% (45253/50000)
Test epoch: 63	Loss: 0.535 | Acc: 82.64% (8264/10000)

Epoch: 64
Train epoch: 64	Loss: 0.276 | Acc: 90.47% (45233/50000)
Test epoch: 64	Loss: 0.399 | Acc: 86.66% (8666/10000)

Epoch: 65
Train epoch: 65	Loss: 0.268 | Acc: 90.93% (45466/50000)
Test epoch: 65	Loss: 0.390 | Acc: 87.17% (8717/10000)
Saving..

Epoch: 66
Train epoch: 66	Loss: 0.270 | Acc: 90.91% (45456/50000)
Test epoch: 66	Loss: 0.416 | Acc: 86.65% (8665/10000)

Epoch: 67
Train epoch: 67	Loss: 0.266 | Acc: 90.93% (45464/50000)
Test epoch: 67	Loss: 0.462 | Acc: 85.09% (8509/10000)

Epoch: 68
Train epoch: 68	Loss: 0.268 | Acc: 90.75% (45374/50000)
Test epoch: 68	Loss: 0.367 | Acc: 88.04% (8804/10000)
Saving..

Epoch: 69
Train epoch: 69	Loss: 0.263 | Acc: 91.08% (45541/50000)
Test epoch: 69	Loss: 0.495 | Acc: 84.02% (8402/10000)

Epoch: 70
Train epoch: 70	Loss: 0.264 | Acc: 90.98% (45491/50000)
Test epoch: 70	Loss: 0.398 | Acc: 87.31% (8731/10000)

Epoch: 71
Train epoch: 71	Loss: 0.250 | Acc: 91.52% (45758/50000)
Test epoch: 71	Loss: 0.409 | Acc: 86.24% (8624/10000)

Epoch: 72
Train epoch: 72	Loss: 0.258 | Acc: 91.12% (45561/50000)
Test epoch: 72	Loss: 0.344 | Acc: 88.69% (8869/10000)
Saving..

Epoch: 73
Train epoch: 73	Loss: 0.254 | Acc: 91.31% (45656/50000)
Test epoch: 73	Loss: 0.362 | Acc: 88.23% (8823/10000)

Epoch: 74
Train epoch: 74	Loss: 0.246 | Acc: 91.48% (45741/50000)
Test epoch: 74	Loss: 0.411 | Acc: 86.60% (8660/10000)

Epoch: 75
Train epoch: 75	Loss: 0.251 | Acc: 91.35% (45676/50000)
Test epoch: 75	Loss: 0.485 | Acc: 85.07% (8507/10000)

Epoch: 76
Train epoch: 76	Loss: 0.246 | Acc: 91.70% (45851/50000)
Test epoch: 76	Loss: 0.438 | Acc: 85.85% (8585/10000)

Epoch: 77
Train epoch: 77	Loss: 0.246 | Acc: 91.70% (45850/50000)
Test epoch: 77	Loss: 0.412 | Acc: 86.70% (8670/10000)

Epoch: 78
Train epoch: 78	Loss: 0.241 | Acc: 91.75% (45876/50000)
Test epoch: 78	Loss: 0.353 | Acc: 88.19% (8819/10000)

Epoch: 79
Train epoch: 79	Loss: 0.237 | Acc: 91.82% (45911/50000)
Test epoch: 79	Loss: 0.433 | Acc: 86.03% (8603/10000)

Epoch: 80
Train epoch: 80	Loss: 0.244 | Acc: 91.69% (45844/50000)
Test epoch: 80	Loss: 0.337 | Acc: 89.01% (8901/10000)
Saving..

Epoch: 81
Train epoch: 81	Loss: 0.231 | Acc: 92.09% (46046/50000)
Test epoch: 81	Loss: 0.401 | Acc: 87.05% (8705/10000)

Epoch: 82
Train epoch: 82	Loss: 0.232 | Acc: 92.14% (46068/50000)
Test epoch: 82	Loss: 0.378 | Acc: 87.78% (8778/10000)

Epoch: 83
Train epoch: 83	Loss: 0.231 | Acc: 92.10% (46048/50000)
Test epoch: 83	Loss: 0.514 | Acc: 84.16% (8416/10000)

Epoch: 84
Train epoch: 84	Loss: 0.233 | Acc: 91.97% (45985/50000)
Test epoch: 84	Loss: 0.363 | Acc: 88.33% (8833/10000)

Epoch: 85
Train epoch: 85	Loss: 0.225 | Acc: 92.29% (46144/50000)
Test epoch: 85	Loss: 0.355 | Acc: 88.71% (8871/10000)

Epoch: 86
Train epoch: 86	Loss: 0.219 | Acc: 92.53% (46264/50000)
Test epoch: 86	Loss: 0.489 | Acc: 84.62% (8462/10000)

Epoch: 87
Train epoch: 87	Loss: 0.217 | Acc: 92.51% (46254/50000)
Test epoch: 87	Loss: 0.414 | Acc: 87.07% (8707/10000)

Epoch: 88
Train epoch: 88	Loss: 0.219 | Acc: 92.44% (46218/50000)
Test epoch: 88	Loss: 0.339 | Acc: 88.46% (8846/10000)

Epoch: 89
Train epoch: 89	Loss: 0.217 | Acc: 92.67% (46334/50000)
Test epoch: 89	Loss: 0.340 | Acc: 88.87% (8887/10000)

Epoch: 90
Train epoch: 90	Loss: 0.213 | Acc: 92.70% (46349/50000)
Test epoch: 90	Loss: 0.375 | Acc: 88.13% (8813/10000)

Epoch: 91
Train epoch: 91	Loss: 0.214 | Acc: 92.66% (46332/50000)
Test epoch: 91	Loss: 0.463 | Acc: 85.57% (8557/10000)

Epoch: 92
Train epoch: 92	Loss: 0.210 | Acc: 92.86% (46428/50000)
Test epoch: 92	Loss: 0.393 | Acc: 87.40% (8740/10000)

Epoch: 93
Train epoch: 93	Loss: 0.205 | Acc: 92.93% (46464/50000)
Test epoch: 93	Loss: 0.389 | Acc: 87.72% (8772/10000)

Epoch: 94
Train epoch: 94	Loss: 0.203 | Acc: 93.11% (46557/50000)
Test epoch: 94	Loss: 0.324 | Acc: 89.42% (8942/10000)
Saving..

Epoch: 95
Train epoch: 95	Loss: 0.198 | Acc: 93.30% (46650/50000)
Test epoch: 95	Loss: 0.365 | Acc: 88.54% (8854/10000)

Epoch: 96
Train epoch: 96	Loss: 0.200 | Acc: 93.17% (46585/50000)
Test epoch: 96	Loss: 0.304 | Acc: 90.38% (9038/10000)
Saving..

Epoch: 97
Train epoch: 97	Loss: 0.195 | Acc: 93.41% (46706/50000)
Test epoch: 97	Loss: 0.339 | Acc: 88.90% (8890/10000)

Epoch: 98
Train epoch: 98	Loss: 0.194 | Acc: 93.33% (46663/50000)
Test epoch: 98	Loss: 0.393 | Acc: 87.77% (8777/10000)

Epoch: 99
Train epoch: 99	Loss: 0.193 | Acc: 93.42% (46710/50000)
Test epoch: 99	Loss: 0.310 | Acc: 89.79% (8979/10000)

Epoch: 100
Train epoch: 100	Loss: 0.189 | Acc: 93.44% (46720/50000)
Test epoch: 100	Loss: 0.330 | Acc: 89.26% (8926/10000)

Epoch: 101
Train epoch: 101	Loss: 0.187 | Acc: 93.57% (46783/50000)
Test epoch: 101	Loss: 0.395 | Acc: 87.89% (8789/10000)

Epoch: 102
Train epoch: 102	Loss: 0.181 | Acc: 93.89% (46943/50000)
Test epoch: 102	Loss: 0.403 | Acc: 87.24% (8724/10000)

Epoch: 103
Train epoch: 103	Loss: 0.178 | Acc: 93.81% (46903/50000)
Test epoch: 103	Loss: 0.398 | Acc: 87.67% (8767/10000)

Epoch: 104
Train epoch: 104	Loss: 0.174 | Acc: 94.06% (47032/50000)
Test epoch: 104	Loss: 0.357 | Acc: 88.63% (8863/10000)

Epoch: 105
Train epoch: 105	Loss: 0.177 | Acc: 93.95% (46973/50000)
Test epoch: 105	Loss: 0.317 | Acc: 89.84% (8984/10000)

Epoch: 106
Train epoch: 106	Loss: 0.174 | Acc: 94.00% (47002/50000)
Test epoch: 106	Loss: 0.344 | Acc: 88.69% (8869/10000)

Epoch: 107
Train epoch: 107	Loss: 0.169 | Acc: 94.24% (47119/50000)
Test epoch: 107	Loss: 0.339 | Acc: 89.44% (8944/10000)

Epoch: 108
Train epoch: 108	Loss: 0.161 | Acc: 94.48% (47242/50000)
Test epoch: 108	Loss: 0.325 | Acc: 89.71% (8971/10000)

Epoch: 109
Train epoch: 109	Loss: 0.166 | Acc: 94.32% (47161/50000)
Test epoch: 109	Loss: 0.318 | Acc: 89.82% (8982/10000)

Epoch: 110
Train epoch: 110	Loss: 0.162 | Acc: 94.54% (47268/50000)
Test epoch: 110	Loss: 0.305 | Acc: 89.92% (8992/10000)

Epoch: 111
Train epoch: 111	Loss: 0.156 | Acc: 94.61% (47305/50000)
Test epoch: 111	Loss: 0.357 | Acc: 89.22% (8922/10000)

Epoch: 112
Train epoch: 112	Loss: 0.158 | Acc: 94.53% (47267/50000)
Test epoch: 112	Loss: 0.445 | Acc: 86.81% (8681/10000)

Epoch: 113
Train epoch: 113	Loss: 0.153 | Acc: 94.70% (47350/50000)
Test epoch: 113	Loss: 0.318 | Acc: 90.00% (9000/10000)

Epoch: 114
Train epoch: 114	Loss: 0.146 | Acc: 95.08% (47540/50000)
Test epoch: 114	Loss: 0.452 | Acc: 86.63% (8663/10000)

Epoch: 115
Train epoch: 115	Loss: 0.145 | Acc: 95.00% (47501/50000)
Test epoch: 115	Loss: 0.360 | Acc: 88.82% (8882/10000)

Epoch: 116
Train epoch: 116	Loss: 0.146 | Acc: 94.99% (47493/50000)
Test epoch: 116	Loss: 0.293 | Acc: 90.79% (9079/10000)
Saving..

Epoch: 117
Train epoch: 117	Loss: 0.142 | Acc: 95.05% (47527/50000)
Test epoch: 117	Loss: 0.345 | Acc: 89.31% (8931/10000)

Epoch: 118
Train epoch: 118	Loss: 0.132 | Acc: 95.52% (47759/50000)
Test epoch: 118	Loss: 0.303 | Acc: 90.57% (9057/10000)

Epoch: 119
Train epoch: 119	Loss: 0.134 | Acc: 95.41% (47706/50000)
Test epoch: 119	Loss: 0.341 | Acc: 89.77% (8977/10000)

Epoch: 120
Train epoch: 120	Loss: 0.132 | Acc: 95.47% (47735/50000)
Test epoch: 120	Loss: 0.360 | Acc: 88.74% (8874/10000)

Epoch: 121
Train epoch: 121	Loss: 0.125 | Acc: 95.65% (47827/50000)
Test epoch: 121	Loss: 0.315 | Acc: 90.00% (9000/10000)

Epoch: 122
Train epoch: 122	Loss: 0.127 | Acc: 95.60% (47800/50000)
Test epoch: 122	Loss: 0.302 | Acc: 90.61% (9061/10000)

Epoch: 123
Train epoch: 123	Loss: 0.123 | Acc: 95.80% (47898/50000)
Test epoch: 123	Loss: 0.303 | Acc: 90.95% (9095/10000)
Saving..

Epoch: 124
Train epoch: 124	Loss: 0.125 | Acc: 95.81% (47904/50000)
Test epoch: 124	Loss: 0.289 | Acc: 90.73% (9073/10000)

Epoch: 125
Train epoch: 125	Loss: 0.114 | Acc: 96.15% (48076/50000)
Test epoch: 125	Loss: 0.305 | Acc: 90.61% (9061/10000)

Epoch: 126
Train epoch: 126	Loss: 0.107 | Acc: 96.36% (48178/50000)
Test epoch: 126	Loss: 0.300 | Acc: 90.90% (9090/10000)

Epoch: 127
Train epoch: 127	Loss: 0.114 | Acc: 96.13% (48067/50000)
Test epoch: 127	Loss: 0.361 | Acc: 89.14% (8914/10000)

Epoch: 128
Train epoch: 128	Loss: 0.108 | Acc: 96.33% (48164/50000)
Test epoch: 128	Loss: 0.298 | Acc: 90.76% (9076/10000)

Epoch: 129
Train epoch: 129	Loss: 0.104 | Acc: 96.40% (48200/50000)
Test epoch: 129	Loss: 0.289 | Acc: 91.15% (9115/10000)
Saving..

Epoch: 130
Train epoch: 130	Loss: 0.106 | Acc: 96.41% (48203/50000)
Test epoch: 130	Loss: 0.291 | Acc: 90.99% (9099/10000)

Epoch: 131
Train epoch: 131	Loss: 0.098 | Acc: 96.67% (48336/50000)
Test epoch: 131	Loss: 0.299 | Acc: 90.82% (9082/10000)

Epoch: 132
Train epoch: 132	Loss: 0.097 | Acc: 96.77% (48385/50000)
Test epoch: 132	Loss: 0.302 | Acc: 91.28% (9128/10000)
Saving..

Epoch: 133
Train epoch: 133	Loss: 0.091 | Acc: 96.95% (48477/50000)
Test epoch: 133	Loss: 0.295 | Acc: 91.53% (9153/10000)
Saving..

Epoch: 134
Train epoch: 134	Loss: 0.090 | Acc: 96.99% (48496/50000)
Test epoch: 134	Loss: 0.297 | Acc: 91.50% (9150/10000)

Epoch: 135
Train epoch: 135	Loss: 0.084 | Acc: 97.15% (48577/50000)
Test epoch: 135	Loss: 0.305 | Acc: 91.06% (9106/10000)

Epoch: 136
Train epoch: 136	Loss: 0.084 | Acc: 97.15% (48573/50000)
Test epoch: 136	Loss: 0.270 | Acc: 91.77% (9177/10000)
Saving..

Epoch: 137
Train epoch: 137	Loss: 0.084 | Acc: 97.19% (48596/50000)
Test epoch: 137	Loss: 0.265 | Acc: 92.21% (9221/10000)
Saving..

Epoch: 138
Train epoch: 138	Loss: 0.079 | Acc: 97.33% (48667/50000)
Test epoch: 138	Loss: 0.285 | Acc: 91.32% (9132/10000)

Epoch: 139
Train epoch: 139	Loss: 0.076 | Acc: 97.45% (48723/50000)
Test epoch: 139	Loss: 0.324 | Acc: 90.41% (9041/10000)

Epoch: 140
Train epoch: 140	Loss: 0.072 | Acc: 97.64% (48818/50000)
Test epoch: 140	Loss: 0.268 | Acc: 92.18% (9218/10000)

Epoch: 141
Train epoch: 141	Loss: 0.067 | Acc: 97.78% (48891/50000)
Test epoch: 141	Loss: 0.264 | Acc: 92.42% (9242/10000)
Saving..

Epoch: 142
Train epoch: 142	Loss: 0.068 | Acc: 97.74% (48870/50000)
Test epoch: 142	Loss: 0.276 | Acc: 92.16% (9216/10000)

Epoch: 143
Train epoch: 143	Loss: 0.063 | Acc: 97.87% (48937/50000)
Test epoch: 143	Loss: 0.332 | Acc: 91.05% (9105/10000)

Epoch: 144
Train epoch: 144	Loss: 0.060 | Acc: 98.04% (49020/50000)
Test epoch: 144	Loss: 0.281 | Acc: 91.78% (9178/10000)

Epoch: 145
Train epoch: 145	Loss: 0.060 | Acc: 98.04% (49020/50000)
Test epoch: 145	Loss: 0.260 | Acc: 92.31% (9231/10000)

Epoch: 146
Train epoch: 146	Loss: 0.055 | Acc: 98.10% (49052/50000)
Test epoch: 146	Loss: 0.296 | Acc: 91.92% (9192/10000)

Epoch: 147
Train epoch: 147	Loss: 0.054 | Acc: 98.26% (49132/50000)
Test epoch: 147	Loss: 0.263 | Acc: 92.64% (9264/10000)
Saving..

Epoch: 148
Train epoch: 148	Loss: 0.049 | Acc: 98.38% (49190/50000)
Test epoch: 148	Loss: 0.317 | Acc: 91.50% (9150/10000)

Epoch: 149
Train epoch: 149	Loss: 0.048 | Acc: 98.37% (49183/50000)
Test epoch: 149	Loss: 0.247 | Acc: 93.06% (9306/10000)
Saving..

Epoch: 150
Train epoch: 150	Loss: 0.049 | Acc: 98.36% (49179/50000)
Test epoch: 150	Loss: 0.246 | Acc: 92.81% (9281/10000)

Epoch: 151
Train epoch: 151	Loss: 0.041 | Acc: 98.71% (49355/50000)
Test epoch: 151	Loss: 0.243 | Acc: 93.27% (9327/10000)
Saving..

Epoch: 152
Train epoch: 152	Loss: 0.039 | Acc: 98.72% (49359/50000)
Test epoch: 152	Loss: 0.240 | Acc: 93.28% (9328/10000)
Saving..

Epoch: 153
Train epoch: 153	Loss: 0.035 | Acc: 98.87% (49437/50000)
Test epoch: 153	Loss: 0.241 | Acc: 93.24% (9324/10000)

Epoch: 154
Train epoch: 154	Loss: 0.035 | Acc: 98.85% (49423/50000)
Test epoch: 154	Loss: 0.238 | Acc: 93.42% (9342/10000)
Saving..

Epoch: 155
Train epoch: 155	Loss: 0.032 | Acc: 99.00% (49498/50000)
Test epoch: 155	Loss: 0.241 | Acc: 93.47% (9347/10000)
Saving..

Epoch: 156
Train epoch: 156	Loss: 0.027 | Acc: 99.16% (49582/50000)
Test epoch: 156	Loss: 0.242 | Acc: 93.36% (9336/10000)

Epoch: 157
Train epoch: 157	Loss: 0.027 | Acc: 99.16% (49582/50000)
Test epoch: 157	Loss: 0.267 | Acc: 92.70% (9270/10000)

Epoch: 158
Train epoch: 158	Loss: 0.021 | Acc: 99.37% (49683/50000)
Test epoch: 158	Loss: 0.217 | Acc: 94.30% (9430/10000)
Saving..

Epoch: 159
Train epoch: 159	Loss: 0.022 | Acc: 99.31% (49654/50000)
Test epoch: 159	Loss: 0.233 | Acc: 93.66% (9366/10000)

Epoch: 160
Train epoch: 160	Loss: 0.017 | Acc: 99.53% (49763/50000)
Test epoch: 160	Loss: 0.243 | Acc: 93.74% (9374/10000)

Epoch: 161
Train epoch: 161	Loss: 0.017 | Acc: 99.50% (49750/50000)
Test epoch: 161	Loss: 0.252 | Acc: 93.44% (9344/10000)

Epoch: 162
Train epoch: 162	Loss: 0.016 | Acc: 99.52% (49762/50000)
Test epoch: 162	Loss: 0.235 | Acc: 93.89% (9389/10000)

Epoch: 163
Train epoch: 163	Loss: 0.015 | Acc: 99.54% (49769/50000)
Test epoch: 163	Loss: 0.235 | Acc: 94.18% (9418/10000)

Epoch: 164
Train epoch: 164	Loss: 0.011 | Acc: 99.68% (49840/50000)
Test epoch: 164	Loss: 0.225 | Acc: 94.22% (9422/10000)

Epoch: 165
Train epoch: 165	Loss: 0.008 | Acc: 99.78% (49890/50000)
Test epoch: 165	Loss: 0.228 | Acc: 94.06% (9406/10000)

Epoch: 166
Train epoch: 166	Loss: 0.009 | Acc: 99.77% (49887/50000)
Test epoch: 166	Loss: 0.218 | Acc: 94.42% (9442/10000)
Saving..

Epoch: 167
Train epoch: 167	Loss: 0.008 | Acc: 99.77% (49883/50000)
Test epoch: 167	Loss: 0.211 | Acc: 94.63% (9463/10000)
Saving..

Epoch: 168
Train epoch: 168	Loss: 0.007 | Acc: 99.84% (49921/50000)
Test epoch: 168	Loss: 0.211 | Acc: 94.47% (9447/10000)

Epoch: 169
Train epoch: 169	Loss: 0.005 | Acc: 99.88% (49940/50000)
Test epoch: 169	Loss: 0.194 | Acc: 94.99% (9499/10000)
Saving..

Epoch: 170
Train epoch: 170	Loss: 0.005 | Acc: 99.89% (49943/50000)
Test epoch: 170	Loss: 0.199 | Acc: 94.88% (9488/10000)

Epoch: 171
Train epoch: 171	Loss: 0.004 | Acc: 99.94% (49970/50000)
Test epoch: 171	Loss: 0.194 | Acc: 94.86% (9486/10000)

Epoch: 172
Train epoch: 172	Loss: 0.003 | Acc: 99.96% (49980/50000)
Test epoch: 172	Loss: 0.191 | Acc: 94.92% (9492/10000)

Epoch: 173
Train epoch: 173	Loss: 0.003 | Acc: 99.96% (49980/50000)
Test epoch: 173	Loss: 0.182 | Acc: 95.11% (9511/10000)
Saving..

Epoch: 174
Train epoch: 174	Loss: 0.003 | Acc: 99.97% (49985/50000)
Test epoch: 174	Loss: 0.190 | Acc: 94.92% (9492/10000)

Epoch: 175
Train epoch: 175	Loss: 0.002 | Acc: 99.98% (49991/50000)
Test epoch: 175	Loss: 0.181 | Acc: 95.09% (9509/10000)

Epoch: 176
Train epoch: 176	Loss: 0.002 | Acc: 99.98% (49988/50000)
Test epoch: 176	Loss: 0.179 | Acc: 95.10% (9510/10000)

Epoch: 177
Train epoch: 177	Loss: 0.002 | Acc: 99.99% (49993/50000)
Test epoch: 177	Loss: 0.180 | Acc: 95.05% (9505/10000)

Epoch: 178
Train epoch: 178	Loss: 0.002 | Acc: 99.99% (49996/50000)
Test epoch: 178	Loss: 0.176 | Acc: 95.14% (9514/10000)
Saving..

Epoch: 179
Train epoch: 179	Loss: 0.002 | Acc: 99.99% (49993/50000)
Test epoch: 179	Loss: 0.177 | Acc: 95.15% (9515/10000)
Saving..

Epoch: 180
Train epoch: 180	Loss: 0.002 | Acc: 99.99% (49995/50000)
Test epoch: 180	Loss: 0.176 | Acc: 95.22% (9522/10000)
Saving..

Epoch: 181
Train epoch: 181	Loss: 0.002 | Acc: 99.99% (49994/50000)
Test epoch: 181	Loss: 0.175 | Acc: 95.21% (9521/10000)

Epoch: 182
Train epoch: 182	Loss: 0.002 | Acc: 99.98% (49990/50000)
Test epoch: 182	Loss: 0.174 | Acc: 95.17% (9517/10000)

Epoch: 183
Train epoch: 183	Loss: 0.002 | Acc: 99.99% (49996/50000)
Test epoch: 183	Loss: 0.170 | Acc: 95.40% (9540/10000)
Saving..

Epoch: 184
Train epoch: 184	Loss: 0.002 | Acc: 99.99% (49996/50000)
Test epoch: 184	Loss: 0.172 | Acc: 95.43% (9543/10000)
Saving..

Epoch: 185
Train epoch: 185	Loss: 0.002 | Acc: 100.00% (49999/50000)
Test epoch: 185	Loss: 0.171 | Acc: 95.36% (9536/10000)

Epoch: 186
Train epoch: 186	Loss: 0.002 | Acc: 99.99% (49996/50000)
Test epoch: 186	Loss: 0.172 | Acc: 95.43% (9543/10000)

Epoch: 187
Train epoch: 187	Loss: 0.002 | Acc: 99.99% (49997/50000)
Test epoch: 187	Loss: 0.170 | Acc: 95.31% (9531/10000)

Epoch: 188
Train epoch: 188	Loss: 0.002 | Acc: 100.00% (50000/50000)
Test epoch: 188	Loss: 0.171 | Acc: 95.41% (9541/10000)

Epoch: 189
Train epoch: 189	Loss: 0.002 | Acc: 100.00% (49999/50000)
Test epoch: 189	Loss: 0.169 | Acc: 95.37% (9537/10000)

Epoch: 190
Train epoch: 190	Loss: 0.002 | Acc: 99.99% (49995/50000)
Test epoch: 190	Loss: 0.170 | Acc: 95.35% (9535/10000)

Epoch: 191
Train epoch: 191	Loss: 0.002 | Acc: 99.99% (49996/50000)
Test epoch: 191	Loss: 0.170 | Acc: 95.39% (9539/10000)

Epoch: 192
Train epoch: 192	Loss: 0.002 | Acc: 100.00% (49998/50000)
Test epoch: 192	Loss: 0.171 | Acc: 95.32% (9532/10000)

Epoch: 193
Train epoch: 193	Loss: 0.002 | Acc: 100.00% (49998/50000)
Test epoch: 193	Loss: 0.170 | Acc: 95.33% (9533/10000)

Epoch: 194
Train epoch: 194	Loss: 0.002 | Acc: 99.99% (49997/50000)
Test epoch: 194	Loss: 0.170 | Acc: 95.32% (9532/10000)

Epoch: 195
Train epoch: 195	Loss: 0.002 | Acc: 99.99% (49997/50000)
Test epoch: 195	Loss: 0.170 | Acc: 95.36% (9536/10000)

Epoch: 196
Train epoch: 196	Loss: 0.002 | Acc: 100.00% (49999/50000)
Test epoch: 196	Loss: 0.170 | Acc: 95.32% (9532/10000)

Epoch: 197
Train epoch: 197	Loss: 0.002 | Acc: 100.00% (49999/50000)
Test epoch: 197	Loss: 0.171 | Acc: 95.32% (9532/10000)

Epoch: 198
Train epoch: 198	Loss: 0.002 | Acc: 100.00% (49999/50000)
Test epoch: 198	Loss: 0.169 | Acc: 95.44% (9544/10000)
Saving..

Epoch: 199
Train epoch: 199	Loss: 0.002 | Acc: 99.99% (49994/50000)
Test epoch: 199	Loss: 0.170 | Acc: 95.40% (9540/10000)
"""

# Regular expression to match accuracy values
pattern = r"Acc: (\d+\.\d+%)"

# Find all matches in the text
matches = re.findall(pattern, text)

# Convert the matches (strings) to float values for better analysis
accuracy_values = [float(match.replace("%", "")) for match in matches]

# Output the results
print("Train Accuracies:")
for i, acc in enumerate(accuracy_values[::2]):
    print(f"{acc}".replace(".", ","))

print("\nTest Accuracies:")
for i, acc in enumerate(accuracy_values[1::2]):
    print(f"{acc}".replace(".", ","))
