# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-09-17

### Bug Fixes

- Support tuning on Python 3.10 ([28f1e05](https://github.com/delyan-boychev/disentangled-flash/commit/28f1e05ad212a1ecee4eb4582838c13fee4fdbb4))
- Make CUDA benchmark parity capacity safe ([2f236ad](https://github.com/delyan-boychev/disentangled-flash/commit/2f236adbb48f6e7f180a921b50c9071f5eea6f69))
- Benchmark the 4096 sequence family ([04f52ba](https://github.com/delyan-boychev/disentangled-flash/commit/04f52ba48945ed7cc9989d6f7130a3f5690d78ac))

### Features

- Tune bounded kernels through 8192 ([3d4db57](https://github.com/delyan-boychev/disentangled-flash/commit/3d4db575f3573fc19f409371931d6319e93541f3))
- Make tuning profiles compiler safe ([6a44316](https://github.com/delyan-boychev/disentangled-flash/commit/6a44316ba10f05ac9d920291706e192845528d26))

### Miscellaneous Tasks

- Add reproducible DeBERTa encoder matrix ([31afb88](https://github.com/delyan-boychev/disentangled-flash/commit/31afb8882efd54389b564127cb338bdffba9e046))
- Define MNLI parity by classification decisions ([f67c41a](https://github.com/delyan-boychev/disentangled-flash/commit/f67c41a833f747cf5420ae63c13f7e3ed8167d77))

### Performance

- Reduce packed encoder overhead ([3056c53](https://github.com/delyan-boychev/disentangled-flash/commit/3056c53c3226963523fe568a65a6eb8eb75de9a1))

### Bench

- Distinguish supported encoder layouts ([be2d297](https://github.com/delyan-boychev/disentangled-flash/commit/be2d297f08ac3ca4b7c197558916fd33cc30ead6))
- Publish H200 results and FlashDeBERTa MNLI ([3082ac6](https://github.com/delyan-boychev/disentangled-flash/commit/3082ac64de0d98efcc6cfaa6164d9be2b0b1a19d))
- Expose strict profile-only MNLI runs ([e5245cc](https://github.com/delyan-boychev/disentangled-flash/commit/e5245cc341bfb7787d805efa26847037164bdf91))
- Evaluate real GLUE MNLI matrix ([a4b2801](https://github.com/delyan-boychev/disentangled-flash/commit/a4b280145245456185614acbc5dc31e85e9429d9))

## [0.1.4] - 2026-09-15

### Bug Fixes

- Pass strides to configured Triton launcher ([3e7fd11](https://github.com/delyan-boychev/disentangled-flash/commit/3e7fd113fcf02aabf6b32c9d007f4ec0f143f477))
- Use integer mask for packed convolution ([73ec060](https://github.com/delyan-boychev/disentangled-flash/commit/73ec060339a4b047b0f72f6716c11ad38ca42052))

### Features

- Add single-launch packed Triton attention ([b62c2ff](https://github.com/delyan-boychev/disentangled-flash/commit/b62c2fffc261b445563288dc0314acc654070ecc))

### Bench

- Default MNLI parity to packed inference ([5507648](https://github.com/delyan-boychev/disentangled-flash/commit/5507648f36de4b095674f6781ec34ff4bf534835))

## [0.1.3] - 2026-09-15

### Documentation

- Align README with runtime tuning families ([0a8fe31](https://github.com/delyan-boychev/disentangled-flash/commit/0a8fe311e7a9ca58d6120d9919bcbce711b3b3b5))

## [0.1.2] - 2026-09-01

### Bug Fixes

- Standardize torch backend name ([cfbcde5](https://github.com/delyan-boychev/disentangled-flash/commit/cfbcde5a7f92fbe267f0f9ff2abb1f4ed178be28))

## [0.1.1] - 2026-09-01

### Documentation

- Add PyPI version badge to README.md ([141b420](https://github.com/delyan-boychev/disentangled-flash/commit/141b420020a1889886bf0eb76c0c32d7cfca453a))

## [0.1.0] - 2026-08-21

### Bug Fixes

- *(release)* Add none option to bump_version.py ([d5ab43c](https://github.com/delyan-boychev/disentangled-flash/commit/d5ab43cd00facbffa02d947e74c229db63c349ac))

### Miscellaneous Tasks

- Use RELEASE_TOKEN in release checkout step ([427d71c](https://github.com/delyan-boychev/disentangled-flash/commit/427d71c6dbf4f1c8f25d49592db6c3ff1760d214))
- Merge pull request #2 from delyan-boychev/release-pipeline-updates

ci: use RELEASE_TOKEN in release workflow ([0b940fa](https://github.com/delyan-boychev/disentangled-flash/commit/0b940fa2ab7b97fb51731746d344a5183f801a91))
- Remove emojis from changelog group headers and fix github_repo variable in cliff.toml ([5019fe3](https://github.com/delyan-boychev/disentangled-flash/commit/5019fe3f69d0c25fb5f8a7848e9457b6876dbf68))
- Merge pull request #3 from delyan-boychev/fix-release-changelog

ci: remove emojis and fix rendering in cliff.toml ([d4ccb26](https://github.com/delyan-boychev/disentangled-flash/commit/d4ccb26bb48cf8424c117249d6f759e1a0154e28))
- Merge PyPI publishing back into release.yml and remove pypi-publish.yml ([c3d103f](https://github.com/delyan-boychev/disentangled-flash/commit/c3d103f898c40ba5e2c05892e5cba3da80fc6910))
- Merge pull request #4 from delyan-boychev/fix-pypi-publishing

ci: merge PyPI publishing into release workflow ([716b5e8](https://github.com/delyan-boychev/disentangled-flash/commit/716b5e84ef82f859d49f65cefaa8e693fea3f12c))
- Reset version back to 0.1.0 for first release ([f052316](https://github.com/delyan-boychev/disentangled-flash/commit/f052316e319be1e3243712af272735b5ed5c0db3))
- Reset version back to 0.1.0 in code files ([3fedad3](https://github.com/delyan-boychev/disentangled-flash/commit/3fedad36360d6ddc558035b3cd354115e17c1acf))
- Reset version and remove changelog for clean release ([4e91841](https://github.com/delyan-boychev/disentangled-flash/commit/4e9184107b7d041e98fa2310a81cfce5d7e68b32))

<!-- generated by git-cliff -->
