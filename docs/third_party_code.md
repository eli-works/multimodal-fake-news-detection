# Third-Party Code Notes

This repository contains or references third-party baseline implementations and external research code for comparison purposes.

## Purpose

The third-party code is kept here to:

1. document baseline sources used during the project
2. preserve the comparison context for experiments
3. support paper-level reproducibility discussion

It should not be interpreted as original code written for the proposed model unless explicitly stated.

## Relevant directory

```text
code/04_第三方基线与参考实现/
```

Current subdirectories include:

- `EANN基线实现`
- `M3FEND参考实现`
- `多模态假新闻系统参考实现`

## Recommended attribution practice

For each third-party subdirectory, the public repository should eventually record:

1. original project or paper title
2. original repository or publication URL
3. upstream license
4. whether the code is copied, adapted, or only referenced
5. what local modifications were made for this project

## Important publication reminder

Before broad public release, double-check that:

- the included third-party code is allowed to be redistributed
- the original license text is preserved when required
- local modifications are clearly marked
- benchmark results derived from adapted code are documented as adapted runs

## Suggested future cleanup

If you want the repository to look more polished for public readers, a good next step is to add a short README inside each third-party baseline directory that answers:

- where it came from
- what it was used for
- whether it was modified
- whether readers should run it directly or only treat it as reference material
