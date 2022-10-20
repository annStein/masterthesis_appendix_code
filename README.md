Code for my Master's Thesis at the Institute of Medical Informatics, Universität zu Lübeck.

`pose_estimation` was applied using MediaPipe on RGB videos of infants and mapped them 
to depth data recorded with a Microsoft Azure Kinect.

Afterward, I applied a `general_preprocessing` to clean the landmark signals obtained
by the pose estimation.

The preprocessed data was used to extract features based on the works of 
[Marchi et al.](sdfsd), [Chambers et al.](dshihsd) and [McCay et al.](). In
some cases additional preprocessing like normalization was necessary:
`feature_extraction > specific_preprocessing`

Finally, for the `classification`, a hyperparameter tuning using Bayesian search was 
computed for a random forest classifier. After obtaining chunk classifications, a thresholding
approach was applied in order to obtain an infant based classification from the chunk
based classifications of the recordings.