# Developers

## Clone a Branch to Another Repository

For clarity, this example demonstrates how to create a clone of the `master` branch from Inria's GitLab to ROMI's GitHub repository.

### Initial Setup (First Time)

1. Create an empty repository on ROMI's GitHub.
2. If not done yet, clone the original GitLab repository.
3. Add a remote named `romi` pointing to the new empty GitHub repository:
   ```shell
   git remote add romi https://github.com/romi/spectral_clustering.git
   ```

### Updating the `master` Branch from the Original Repository

4. Push the local changes from the original `master` branch to ROMI's GitHub repository (execute this from the repository root):
   ```shell
   git push romi master
   ```

## Testing Documentation Locally

To preview and test the MkDocs documentation locally, make sure you have the necessary tools installed on your system.
You can do this by installing the `'doc'` optional dependencies specified in the `pyproject.toml` file as follows:

```shell
python -m pip install -e '.[doc]'
```

Next, navigate to the root directory of your project (where the `mkdocs.yml` file is located) and run the following command:

```shell
python docs/assets/scripts/gen_ref_pages.py
mkdocs serve
```

This command starts a local development server, which can be accessed by opening `http://127.0.0.1:8000/` in your web browser.
Any updates made to the documentation files will automatically refresh in your browser.
