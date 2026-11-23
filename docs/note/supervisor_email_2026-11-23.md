Subject: POGS reproduction: progress report and two questions

Dear Professor [Name],

Please find attached a short progress report on the POGS re-implementation, measured against the requirements you set.

In summary, requirements O1 (object grouping), O2 (code structure and environment) and O3 (recovery from step-wise perturbation up to 30°) are complete. O5 (model latency) and O6 (5 Hz tracking) are partly met. O4, the 80% in-grasp recovery rate, is not met: the rate is 4 to 28% under our pre-registered rule and 34 to 78% under the paper's own rule. Our +30° setting matches the paper's 30° result (39 versus 40 of 50).

We traced the shortfall to a systematic roll error of about 10° about the drill bit. Its cause is a mechanism in the DINO loss. The background pixels around the object, compared with an empty render, act as an implicit silhouette term. This term provides most of the tracker's ability to recover, and shadows and viewpoint bias it. We tested six corrections and none passed the criterion we set in advance.

I would be grateful for your guidance on two questions:

1. Would a mechanism paper on feature-field tracking, built on this finding, be an acceptable outcome? If so, which venue would you consider realistic?

2. May the tracker use a known gripper model (commanded pose and mesh) to exclude gripper pixels? The POGS authors list gripper masking as future work. This would allow one pre-registered test of an explicit, shadow-free silhouette term, capped at six GPU hours, with the same exclusion applied to the baseline.

I would be happy to discuss the results at your convenience.

Kind regards,
[Your name]
