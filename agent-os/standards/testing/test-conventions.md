# Test Conventions

## Organization

Tests live in `test-suite/<rule_name>/`, mirroring `regeln/<rule_name>/`.

## File Naming

```
NN_descriptive_name_flag.txt
NN_descriptive_name_pass.txt
```

- `NN`: two-digit number for ordering (01, 02, ...)
- `_flag`: rule should detect a violation
- `_pass`: rule should find no violations

## File Format

```
# expected: flag|pass
# description: What this test checks

German text to test goes here.
```

- `# expected:` and `# description:` are parsed by the test runner
- Blank line separates metadata from test text
- One test case per file

## Running Tests

```bash
python test-suite/test_runner.py              # all tests
python test-suite/test_runner.py <rule_name>  # single rule
python test-suite/test_runner.py -v           # verbose
```

Always run `python test-suite/test_runner.py <rule_name>` after modifying a rule.
