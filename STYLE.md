# C++ Coding Standards

All C++ code in this repository follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

## Key Formatting Rules
* **Indentation:** 2 spaces. No tabs.
* **Line Length:** Maximum 80 characters.
* **Braces:** Open braces `{` go on the same line as the statement.
* **Pointers:** The `*` stays with the type (e.g., `int* x`, not `int *x`).
* **Includes:** Organized alphabetically to prevent merge conflicts.

## Automated Formatting
We use `clang-format` to maintain consistency. You can format your code locally before committing:

```bash
clang-format -i -style=file <filename>
