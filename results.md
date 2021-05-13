# Notable results

#### Short-Time Fourier Transform Spectogram
-----------------------------------------------
- 150 epochs, 1 fold, 256 batch size, SGD with 0.01 learning rate and 0.8 momentum, dataset came directly from calculations/no images.\
Assumed all 15 frames are an onset... Wrong.\
Accuracy: 0.9039462347825368 (+- 0.0)\
Loss: 0.22193499594926835\
Precision: 0.9204497381051382\
Recall: 0.8995434701442718\
F-Measure: 0.9098765289403388\
True Positives: 294587 True Negatives: 253552\
False Positives: 25347 False Negatives: 32898\
-----------------------------------------------
From here onwards, 5 of the 15 frames are an onset (50ms).\
Same as above, but 100 epochs and considering only 5 of the 15 frames are an onset, bins 100.\
Accuracy: 0.909080730676651 (+- 0.0)\
Loss: 0.22308972865343094\
Precision: 0.8458883762359619\
Recall: 0.7219544041156769\
F-Measure: 0.7790230580094823\
True Positives: 97306 True Negatives: 453714\
False Positives: 17633 False Negatives: 37475
-----------------------------------------------
Same as above but 80 frequency bins.\
Accuracy: 0.8913271355628968 (+- 0.0)\
Loss: 0.27245558977127077\
Precision: 0.8273037481307983\
Recall: 0.6452908682823181\
F-Measure: 0.7250489007828536\
True Positives: 87002 True Negatives: 453543\
False Positives: 18081 False Negatives: 47824

-----------------------------------------------

#### Wavelet Transform Scaleogram
-----------------------------------------------
-----------------------------------------------


#### Constant-Q Transform Spectogram
-----------------------------------------------
25 epochs, 1 fold, 256 batch size\
SGD with 0.01 learning rate and 0.8 momentum\
Dataset came directly from calculations/no images\
Accuracy: 0.877650933265686 (+- 0.0)\
Loss: 0.2761280697584152\
Precision: 0.9058214712142945\
Recall: 0.8629412865638733\
F-Measure: 0.8838616058930818\
True Positives: 282522 True Negatives: 249449\
False Positives: 29287 False Negatives: 44872\
-----------------------------------------------

#### Constant-Q Transform Chromagram
-----------------------------------------------
25 epochs, 1 fold, 256 batch size\
SGD with 0.01 learning rate and 0.8 momentum\
Dataset came directly from calculations/no images\
Accuracy: 0.8420165300369262 (+- 0.0)\
Loss: 0.3509286391735077\
Precision: 0.8486549186706543\
Recall: 0.8611823081970215\
F-Measure: 0.8548727214957401\
True Positives: 281946 True Negatives: 228426\
False Positives: 50310 False Negatives: 45448
-----------------------------------------------
