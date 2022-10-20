Code for my Master's Thesis at the Institute of Medical Informatics, Universität zu Lübeck.

`pose_estimation` was applied using MediaPipe on RGB videos of infants and mapped them 
to depth data recorded with a Microsoft Azure Kinect.

Afterward, I applied a `general_preprocessing` to clean the landmark signals obtained
by the pose estimation.

The preprocessed data was used to extract features based on the works of 
[Marchi et al.](https://www.researchgate.net/publication/331867047_Automated_pose_estimation_captures_key_aspects_of_General_Movements_at_8-17_weeks_from_conventional_videos), [Chambers et al.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9214853) and [McCay et al.](https://www.researchgate.net/publication/336331058_Establishing_Pose_Based_Features_Using_Histograms_for_the_Detection_of_Abnormal_Infant_Movements). In
some cases additional preprocessing like normalization was necessary:
`feature_extraction > specific_preprocessing`

Finally, for the `classification`, a hyperparameter tuning using Bayesian search was 
computed for a random forest classifier. After obtaining chunk classifications, a thresholding
approach was applied in order to obtain an infant based classification from the chunk
based classifications of the recordings.
