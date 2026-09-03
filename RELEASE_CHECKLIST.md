# Release Checklist

## Ready in this staging snapshot

- [x] llama.cpp base revision recorded
- [x] patch exported
- [x] benchmark runners copied
- [x] aggregate Decode, context, Prefill, and negative-result CSVs copied
- [x] model weights and session state excluded
- [x] raw prompt/token/routing payloads excluded
- [x] evidence labels documented
- [x] license inventory drafted
- [x] README, architecture, limitations, and reproduction docs drafted
- [x] publication drafts stored locally
- [x] clean out-of-tree configure and build completed locally

## Maintainer decisions before publication

- [ ] choose copyright holder and license for original standalone files
- [ ] run a second-machine reproduction
- [ ] run multi-prompt Prefill quality evaluation
- [ ] decide whether benchmark-max or safe-default config leads the README
- [ ] review hardware identifiers and result precision
- [ ] inspect the final git diff
- [ ] select repository name and remote
- [ ] approve each public post
