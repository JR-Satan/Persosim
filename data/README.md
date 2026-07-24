# Public Lifelong Data Release

`20user_lifeline.json` / `20user_eval.json` contain 20 personas and 1,000
paired runtime and evaluator tasks. `100user_lifeline.json` /
`100user_eval.json` contain 100 personas and 5,000 paired tasks. Each
`*_lifeline.json` file must be used only with its identically prefixed sibling
`*_eval.json` file.

These aggregate files are researcher artifacts, not direct tested-agent
inputs. A tested agent may receive only the current visible request,
attachment, cues, and retrieved visible transcripts. In particular,
`private_goal`, state-authority data, `need_exact`, checks, metric eligibility,
and view mappings must never be shown to the tested agent.

The 20-user pair is a clean projection of the 100-user release in the legacy
20-persona order. It removes campaign, schema, binding, and semantic-revision
metadata from the prior aggregate release and incorporates the repaired
`sed_003` privacy wording. Regenerate it with:

```bash
python scripts/build_public_20user_projection.py
```

SHA-256 for this clean projection:

| File | SHA-256 |
| --- | --- |
| `20user_lifeline.json` | `dd104cef56b82e06e77257835cbfe9de74e66260567d7bdc35076827cade1dea` |
| `20user_eval.json` | `048b6c09e54f5b649d6031f61a860c8ffdd996b86c828a12fd732ab5d2302a22` |
| `100user_lifeline.json` | `f2dd42748856bf72441d992d55c2bb52b66ab3711b931501390c719cfd46b21b` |
| `100user_eval.json` | `6ee1683a57369cb07a9211500ad59611a68ba2a645770fa4da8c80d27fc04333` |
