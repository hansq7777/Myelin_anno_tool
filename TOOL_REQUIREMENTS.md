# Tool Requirements (Local, Not Committed)

This project needs a few non-Python tools at runtime. These tools should be
installed locally and documented, but their binaries should not be committed to
Git.

## Java Runtime

- Required major version: Java 17
- Verified local build: Temurin OpenJDK 17.0.18+8

Example check:

```bash
java -version
```

## Bio-Formats Command Line Tools

- Required toolset: Bio-Formats CLI (`bfconvert`, `showinf`, etc.)
- Verified local version: 8.4.0
- Build date: 2026-01-14

Example check:

```bash
showinf -version
```

## Notes

- Keep these tools outside source control.
- Use `.gitignore` rules to prevent accidental commits of local tool bundles.
