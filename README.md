# CS249r: Tiny Machine Learning - Collaborative Book

[![All Contributors](https://img.shields.io/github/all-contributors/harvard-edge/cs249r_book?color=ee8449&style=flat-square)](#contributors)

Welcome to the collaborative book repository for students of CS249r: Tiny Machine Learning at Harvard! This repository
contains the source files of chapters and sections written by your peers. We're excited to see your contributions!

---

## Contributing

To get started with your contributions, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harvard-edge/cs249r_book.git
    ```
2. **Navigate to the Repository**:
    ```bash
    cd cs249r_book
    ```
3. **Create a New Branch** for your chapter/section:
    ```bash
    git checkout -b name-of-your-new-branch
    ```
4. Write your chapter/section in Markdown.
5. **Commit Changes to Your Branch**:
    ```bash
    git add .
    git commit -m "Description of your changes"
    ```
6. **Push Your Branch to the Repository**:
    ```bash
    git push origin name-of-your-new-branch
    ```
7. Open a pull request to the `main` branch of the original repository.

The instructors will review your pull request and provide feedback. Once accepted, your changes will be merged into
the `main` branch, and the website will automatically update.

More detailed instructions on the CS249r scribing effort and peer review process can be found [here](https://docs.google.com/document/d/1izDoWwFLnV8XK2FYCl23_9KYL_7EQ5OWLo-PCNUGle0).

---

## Website

The book website is automatically built from the `gh-pages` branch. Changes to `main` will be merged into `gh-pages`
once reviewed.

View the book website at: [https://harvard-edge.github.io/cs249r_book/](https://harvard-edge.github.io/cs249r_book/)

---

## Local Rendering

To render the book locally, you'll need to install `quarto`. Once `quarto` is installed, you can run the following
command to generate the HTML pages:

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

```bash
cd cs249r_book
quarto render
