---
id: f4l9pyyfxzy92fevodtum83
title: '2023-11-01'
desc: ''
updated: 1698825292696
created: 1698820811496
traitIds:
  - journalNote
---
This template was applied using the daily journal schema. Edit the [[templates.daily]] note to change this template.
To create your own schemas to auto-apply templates when they match a hierarchy, follow the [schema tutorial](https://blog.dendron.so/notes/P1DL2uXHpKUCa7hLiFbFA/) to get started.

<!--
Based on the journaling method created by Intelligent Change:
- [Intelligent Change: Our Story](https://www.intelligentchange.com/pages/our-story)
- [The Five Minute Journal](https://www.intelligentchange.com/products/the-five-minute-journal)
-->

## Objectives

- Refactor Flows Tasks, Jobs DTOs and Value Objects
- Build Evaluator Tasks
- Build Experiment Jobs
- Build Evaluation Reporting and Visualization
- Conduct Denoising Experiment

## Refactor Flows Tasks, Jobs DTOs and Value Objects

- Adding builders for jobs and tasks.
- Need an image reader iterator to avoid having to load all images into memory at once.
  - Challenge: avoid circular imports.
  - Since the image reader depends upon the image repository, i'll put them in the same module.
  - The image reader will be obtained from the DI container for Task/Job Builders and loaded into the Job/Task by the builder.
  - Change of plan:
    - Moving image reader to the io package image module. We will inject the repository into the iterator. Why? We need to pass the image selection conditions into the constructor at runtime.
