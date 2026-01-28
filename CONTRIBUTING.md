<!---
Copyright 2020 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribute to 🤗 Transformers.js

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping
others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply ⭐️ the repository to say thank you.

**This guide was heavily inspired by the awesome [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md) and our friends at [transformers](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to 🤗 Transformers.js:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Implement new models.
* Contribute to the examples or to the documentation.

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](#create-a-pull-request) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The 🤗 Transformers.js library is robust and reliable thanks to users who report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the library itself, and not your code.

To create new Issue, please [use on of the templates](https://github.com/huggingface/transformers.js/issues/new/choose) we prepared for you. Most likely the [Bug Report](https://github.com/huggingface/transformers.js/issues/new?template=1_bug-report.yml).

### Do you want a new feature?

If there is a new feature you'd like to see in 🤗 Transformers.js, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it a feature related to something you need for a project? Is it something you worked on and think it could benefit the community? Whatever it is, we'd love to hear about it!
2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.
3. Provide a *code snippet* that demonstrates the features usage.
4. If the feature is related to a paper, please include a link.

If your issue is well written we're already 80% of the way there by the time you create it.

We have added [a template](https://github.com/huggingface/transformers.js/issues/new?template=4_feature-request.yml) to help you get started with your issue.

## Do you want to implement a new model?

New models are constantly released and if you want to implement a new model, please provide use the [template for new model requests](https://github.com/huggingface/transformers.js/issues/new?template=2_new_model.yml)

If you are willing to contribute the model yourself, let us know so we can help you add it to 🤗 Transformers.js!

## Create a Pull Request

Before writing any code, we strongly advise you to search through the existing PRs or
issues to make sure nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to contribute to
🤗 Transformers.js While `git` is not the easiest tool to use, it has the greatest
manual. Type `git --help` in a shell and enjoy! If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

### Prerequisites

You'll need the following tools installed to contribute to 🤗 Transformers.js:

- **[Node.js v18](https://nodejs.org/)** or above
- **[pnpm](https://pnpm.io/)** - Fast, disk space efficient package manager

To install pnpm:
```bash
npm install -g pnpm
```

Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/huggingface/transformers.js) by
   clicking on the **[Fork](https://github.com/huggingface/transformers.js/fork)** button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/transformers.js.git
   cd transformers.js
   ```

 3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

> 🚨 **Do not** work on the `main` branch!

4. Set up a development environment by running the following command:
   ```bash
   pnpm install
   ```
5. Develop the features in your branch.
6. Now you can go to your fork of the repository on GitHub and click on **Pull Request** to open a pull request. Make sure you tick off all the boxes on our [checklist](#pull-request-checklist) below. When you're ready, you can send your changes to the project maintainers for review.
7. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

### Pull request checklist
☐ The pull request title should summarize your contribution.  
☐ If your pull request addresses an issue, please mention the issue number in the pull
request description to make sure they are linked (and people viewing the issue know you
are working on it).  
☐ To indicate a work in progress please prefix the title with `[WIP]`. These are
useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.  
☐ Make sure existing tests pass (`pnpm test`).  
☐ Make sure the build completes successfully (`pnpm build`).  
☐ Make sure your code is [formatted properly with Prettier](#code-formatting) (`pnpm format:check`).  
☐ If adding a new feature, also add tests for it.  
☐ If your changes affect user-facing functionality, update the relevant documentation.

### Tests
We are using [Jest](https://jestjs.io/) to execute unit-tests. All tests can be found in `packages/transformers/tests` and have to end with `.test.js`

Execute all tests
```bash
pnpm test
```

Execute tests for a specific package
```bash
pnpm --filter @huggingface/transformers test
```

Execute a specific test file
```bash
cd packages/transformers
pnpm test -- ./tests/models.test.js
```

### Style guide

#### Code formatting
We use [Prettier](https://prettier.io/) to maintain consistent code formatting across the project. Please ensure your code is formatted before submitting a pull request.

**Format all files:**
```bash
pnpm format
```

**Check formatting without making changes:**
```bash
pnpm format:check
```

**IDE Integration (recommended)**

We recommend setting up Prettier in your IDE to format on save:

**Visual Studio Code:**
1. Install the [Prettier extension](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)
2. Open Settings (Ctrl+, or Cmd+,)
3. Search for "format on save"
4. Enable "Editor: Format On Save"
5. Set Prettier as your default formatter: search for "default formatter" and select "Prettier - Code formatter"

**IntelliJ IDEA / WebStorm:**
1. Go to `Settings` → `Languages & Frameworks` → `JavaScript` → `Prettier`
2. Set the Prettier package path (usually `node_modules/prettier`)
3. Check "On save" under "Run for files"
4. Add file patterns: `{**/*,*}.{js,ts,jsx,tsx,json,css,scss,md}`
5. Click "Apply" and "OK"

## Project Structure

This project uses **pnpm workspaces** to manage multiple packages in a monorepo. Currently, there is one workspace:

- `packages/transformers` - The main Transformers.js library

This structure allows for better organization and makes it easier to add framework-specific integrations in the future.

## How to make changes to transformers.js

### Development workflow

The recommended way to develop and test changes is to use the watch mode build and install from the local package:

1. Start the build in watch mode:
   ```bash
   pnpm dev
   ```
   This will automatically rebuild the library whenever you make changes to the source code.

2. Create a separate test project and install transformers.js from your local development directory:
   ```bash
   mkdir my-test-project
   cd my-test-project
   npm init -y
   npm install file:/path/to/transformers.js/packages/transformers
   ```
   Replace `/path/to/transformers.js` with the actual path to your cloned repository.

3. Make your changes to the transformers.js source code in the main repository. The watch mode will automatically rebuild the library.

4. Test your changes in your test project. The changes will be automatically reflected since the package is linked via the `file:` protocol.

This workflow allows for rapid iteration and testing during development.