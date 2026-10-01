# Changelog

All notable changes to this project will be documented in this file.

## [1.1.1] - 2026-10-01

### Bug Fixes

- Preserve base attention on Triton fallback ([f1606d6](https://github.com/delyan-boychev/disentangled-flash/commit/f1606d623361882b1928193aa62bace5975d470d))

### Documentation

- Add RTX 6000 benchmark results ([b1a02f4](https://github.com/delyan-boychev/disentangled-flash/commit/b1a02f4852e0a90c00b445e6fa5e9387de0dcec7))
- Tighten benchmark documentation ([5db9df6](https://github.com/delyan-boychev/disentangled-flash/commit/5db9df60cfcbca0ef78e92e00b7c86172c0ea3a4))
- Refresh RTX MNLI cache results ([188a0cc](https://github.com/delyan-boychev/disentangled-flash/commit/188a0cce113d460c150dd930681ec63baa57f10f))
- Refresh H200 results and simplify plots ([f538d1e](https://github.com/delyan-boychev/disentangled-flash/commit/f538d1e3485e56edc7487d8e22ac2fc84386e0e9))
- Move FLOP derivation out of figures ([20e545e](https://github.com/delyan-boychev/disentangled-flash/commit/20e545eb432938c2b90b32e8f65fcd4dc15b8b3d))
- Add RTX A6000 benchmark results ([692841d](https://github.com/delyan-boychev/disentangled-flash/commit/692841df8a859bff31e3eee3eaf4620633469cf0))
- Group benchmark results by GPU ([74ac0cd](https://github.com/delyan-boychev/disentangled-flash/commit/74ac0cda5942558bc595c03a7588ff61e400298d))
- Explain attention FLOP accounting ([5beebc6](https://github.com/delyan-boychev/disentangled-flash/commit/5beebc6a3b27ff765c6141f3e4d1881cf85fe321))
- Mark backward as supported ([77ae3fb](https://github.com/delyan-boychev/disentangled-flash/commit/77ae3fb46a0adc3a946347c9f75359ba5da156c0))
- Clarify kernel benchmark layout ([ce64e6d](https://github.com/delyan-boychev/disentangled-flash/commit/ce64e6daf6b25d76e7271050ef72a2928ad711f9))
- Tighten benchmark layout note ([34c8412](https://github.com/delyan-boychev/disentangled-flash/commit/34c8412ba735ce76b153e60d3cb5c93f853cdb42))
- Clarify when packed layout helps ([4bc077c](https://github.com/delyan-boychev/disentangled-flash/commit/4bc077c7b69c3cb381bb0fed832031d2252e78b7))

## [1.1.0] - 2026-09-30

### Documentation

- Simplify throughput figure labels ([76fdf02](https://github.com/delyan-boychev/disentangled-flash/commit/76fdf023703b9c3717e2f0d8f60d102230961e5d))
- Restore three-pass throughput figure ([7863671](https://github.com/delyan-boychev/disentangled-flash/commit/78636714d12be96c4bed9c42840c6289120b2783))

### Miscellaneous Tasks

- Default to a portable heuristic instead of runtime autotuning

Pick launch configs from saved profiles and, on a miss, from a heuristic
that depends only on the workload and on limits the GPU reports (shared
memory per block, SM count). Configs that don't fit are dropped before
compiling, and tiles are split when a launch would leave SMs idle. The
same resolver runs under torch.compile, where it is evaluated once while
tracing and recompiles only per length family. mode="heuristic" ignores
profiles; mode="autotune" keeps runtime autotuning.

The tuner's default standard preset now measures only the heuristic
config and its nearest candidates on a few representative shapes; the
previous full matrix is the exhaustive preset. --verbose prints every
measured candidate.

Training now projects Q/K/V with one GEMM over the original parameters.

The kernel fingerprint ignores module classes and imports. The bundled
H200 profile's fingerprint was updated after checking that the kernel
and launch code is unchanged. ([73c88ae](https://github.com/delyan-boychev/disentangled-flash/commit/73c88aec0822267e98337bb5b1d4f43b5cc61ba8))

### Bench

- Finalize release performance results ([c55d83a](https://github.com/delyan-boychev/disentangled-flash/commit/c55d83a84b4b218eae33655b48ba38e86052fd05))

## [1.0.0] - 2026-09-29

### Bug Fixes

- Fix benchmark worker variant scope ([46db32c](https://github.com/delyan-boychev/disentangled-flash/commit/46db32c01178c8880582ed7fdf9896c209c078f1))
- Mask padded score gradients in backward ([14505a1](https://github.com/delyan-boychev/disentangled-flash/commit/14505a164df44d9b3d49503fc5e4db855c8f8be2))
- Standardize torch backend name ([703a180](https://github.com/delyan-boychev/disentangled-flash/commit/703a1805f232ac32df1261493d383ae1bf1ff23b))
- Handle fully masked Triton tiles ([b657e14](https://github.com/delyan-boychev/disentangled-flash/commit/b657e14930ca6d2ef175017b9392cea9ed125334))
- Align training forward padding keyword with shared kernel ([78a16d9](https://github.com/delyan-boychev/disentangled-flash/commit/78a16d926f83a9e5312b2ac863e2124d0147cbcf))
- Preserve packed convolution mask contract ([32a97a8](https://github.com/delyan-boychev/disentangled-flash/commit/32a97a8e7c7f7a9e65ec0b122b132038c182b1c3))

### Documentation

- Describe optimized training support ([1a89ba9](https://github.com/delyan-boychev/disentangled-flash/commit/1a89ba9fd68338b67804b7d0bc35afed74b546f3))
- Publish H200 benchmark and parity results ([10f2691](https://github.com/delyan-boychev/disentangled-flash/commit/10f2691576aff67895a99bc0f69b97cbb0a4f538))
- Report H200 performance and memory results ([c90a07a](https://github.com/delyan-boychev/disentangled-flash/commit/c90a07aadeb09b34f79c40fc7734322dc3faab3e))

### Features

- Add training backward for disentangled attention ([1339a06](https://github.com/delyan-boychev/disentangled-flash/commit/1339a06b40da200429e0816db40c12225eb8bdb1))

### Miscellaneous Tasks

- Detach training validation metrics ([35d835f](https://github.com/delyan-boychev/disentangled-flash/commit/35d835f30cd8a84f717545857db84e0d0ff62665))
- Add fused attention dropout and precision-family tuning

Apply attention-probability dropout inside the fused forward and
backward kernels. A Philox mask keyed per batch/sequence and head is
regenerated in dQ and dK/dV instead of being stored; the softmax
denominator and LSE stay undropped and the output is scaled by 1/(1-p),
as in FlashAttention. The seed is a device tensor saved for backward, so
it follows the CUDA generator and traces under torch.compile. Training
now uses Triton for configs with attention_probs_dropout_prob > 0.

Tune FP16 and BF16 as one half-precision family, like FlashAttention and
FlexAttention, and key training workloads by dropout. Legacy profiles
with separate FP16/BF16 entries still load. FP32 searches a
conservative single-stage space that mirrors runtime autotune pruning.
The standard matrix is now 162 shapes and 1134 phase workloads. ([bf81da7](https://github.com/delyan-boychev/disentangled-flash/commit/bf81da7f9c404855873bb8dbab93494033bee821))
- Fixed bug materilization large matricies ([571af50](https://github.com/delyan-boychev/disentangled-flash/commit/571af50192e6844b2e05ea373dcd08df353b60f5))
- Install optional dependencies required by tests ([b2a5c9f](https://github.com/delyan-boychev/disentangled-flash/commit/b2a5c9f22ae6ea86826d45842a6f5fe6be931259))

### Refactors

- Clarify padding-mask specialization ([ba0e8cb](https://github.com/delyan-boychev/disentangled-flash/commit/ba0e8cb48764a3095d1a93b0e0367d69bb70b771))

### Testing

- Reuse optimized parity loader ([09012cf](https://github.com/delyan-boychev/disentangled-flash/commit/09012cfaa9ce57ca19b14f8e60489af8eef41394))

### Bench

- Add FlashDeBERTa to pretrained MNLI parity test ([ca9b479](https://github.com/delyan-boychev/disentangled-flash/commit/ca9b47914e5572e47aa0f27f814b366e7cdedbba))

### Benchmark

- Add FlashDeBERTa comparison ([0f3df3e](https://github.com/delyan-boychev/disentangled-flash/commit/0f3df3e36af72c7bb548b1d2e0d2fc79baad74ac))

### Lint

- Clean benchmark worker handling ([d8aca6c](https://github.com/delyan-boychev/disentangled-flash/commit/d8aca6c9cc39332e9d81c0d2502d203845c91582))
- Clean inference benchmark error handling ([c349109](https://github.com/delyan-boychev/disentangled-flash/commit/c3491094431a29cf4e71e331be1b369dbc824191))
- Iterate tensor stats mappings with items ([5fbde70](https://github.com/delyan-boychev/disentangled-flash/commit/5fbde709969507d97353945d7920825bbe48b369))

### Validation

- Add normalized parity metrics ([ee72375](https://github.com/delyan-boychev/disentangled-flash/commit/ee723751d0890b4ab0558064bab804f85de5c7ba))
- Add multistep training parity ([0acb440](https://github.com/delyan-boychev/disentangled-flash/commit/0acb44007fc65555d6d88fa41800342f7e2bf0c9))
- Stabilize multistep parity loss ([dffbe85](https://github.com/delyan-boychev/disentangled-flash/commit/dffbe85c7bd4da45a42c7767f40c13eafb33395c))
- Add learnable structured multistep task ([701ee71](https://github.com/delyan-boychev/disentangled-flash/commit/701ee7186a54000e9251eeb265e0509ade5ada45))

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
