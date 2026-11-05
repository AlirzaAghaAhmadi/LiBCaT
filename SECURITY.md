# Security and responsible disclosure

LiBCaT is research software, not medical decision-support software. Do not enter real patient
identifiers or protected health information into issues, sample files, PRs, or demos.

To report a security problem, use GitHub's private vulnerability reporting feature
(if enabled on the repository) rather than creating a public Issue. Do not include secrets.

The code does not require third-party API keys for local retrieval. ClinicalTrials.gov
snapshot acquisition and optional pretrained model downloads require network access.
