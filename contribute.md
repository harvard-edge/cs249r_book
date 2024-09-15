# Guidlines for contributing to the project

The Machine Learning Systems with TinyML project welcomes contributions from everyone. This project is maintained by a community of contributors from around the world. We appreciate your help!

Your contributions are welcome and can encompass a variety of tasks, such as:

- Identifying and reporting any bugs in the examples
- Correcting typographical errors in the documentation
- Contributing additional examples
- Authoring a new chapter
- Suggesting topics for new chapters
- Enhancing the accessibility of the material

If you are unsure about whether a contribution is appropriate, feel free to open an [issue](https://github.com/harvard-edge/cs249r_book/issues) and ask.

## How to contribute

### Open an issue

If there is an open issue for the contribution you would like to make, please comment on the issue to let us know you are working on it. If there is no open issue, please open one to let us know you are working on the contribution.

### Fork the repository

Fork the repository on GitHub and clone your fork to your local machine. We are following GitHub flow for collabration. Please make sure that your main branch is up to date with the upstream main branch before you start working on your contribution.

### Clone the forked repository

```bash
git clone https://github.com/YOUR_USERNAME/cs249r_book.git
```

### Navigate to the repository

```bash
cd cs249r_book
```

### Add the upstream remote

```bash
git remote add upstream https://github.com/harvard-edge/cs249r_book.git
```

Please note that the upstream remote is read-only. You will not be able to push to the upstream remote. You will only be able to push to your forked repository (you will use a Pull Request for merging your code to upstream). However, you will be able to pull from the upstream remote to keep your forked repository up to date with the upstream repository.

### Create a new branch

Create a branch for your contribution. The branch name should start with issue number and be descriptive of the contribution you are making. For example, if you are fixing a typo in the documentation, the branch name could be `iss14-fix-typo-in-documentation`. If you are adding a new example, the branch name could be `iss5-add-new-example`. Following this naming convention will help us keep track of the ongoing contributions.

### Make your changes

Make your changes to the code or documentation. Please make sure that your changes are consistent with the style of the rest of the code or documentation.  

- The `content` directory subfolders that each represent a chapter in the book. Each chapter folder contains the source files and documents to render the book. Any new files should be added to the `content` directory in its appropriate folder. Please create a new folder if needed. Make sure that the path in the `_quarto.yml` file is updated to include the new folder.

- Each chapter folder also include an images folder. The images folder has 4 subfolders: `png`, `pdf`, `svg`, and `jpg`. Please add your images to the appropriate folder. This is important to keep the images organized and to make sure that the images are rendered correctly in the book.

- Update `updates.md` file with a brief description of your contribution. Please include the issue number in the description. For example, `Fix typo in the documentation (issue #14)`. See [keep a changelog](https://keepachangelog.com/en/1.1.0/) for more information on how to write a good changelog.

### Commit your changes

```bash
git add .
git commit -m "your commit message"
```

### Render the book

Please render the book to make sure that your contribution is rendered correctly and do not raise an error or warnings. We are using [quarto](https://quarto.org/docs/get-started/) to render the book.  You can render the book by running the following command in the terminal:

```bash
quarto render
```

### Push your changes to your forked repository

```bash
git push origin your-branch-name
```

### Open a Pull Request (PR)

Please submit the PRs to the `dev`  branch, not `main`.

Open a Pull Request (PR) to merge your changes to the upstream repository. Please add a brief description of your contribution to the PR. Please include the issue number in the description. For example, `Fix typo in the documentation (issue #14)`.

Opening an early PR is encouraged. This will allow us to provide feedback on your contribution and help you improve it. Moreover, Github Actions will run on your PR and will generate the book, so you can download the book and make sure that your contribution is rendered correctly.

- If your PR is a work in progress, please add `[WIP]` to the title of the PR. This will let us know that you are still working on your contribution and that you are not ready for a review or merge yet.

For a more detailed guide on the CS249r documentation process and peer review,
check [here](https://docs.google.com/document/d/1izDoWwFLnV8XK2FYCl23_9KYL_7EQ5OWLo-PCNUGle0).
