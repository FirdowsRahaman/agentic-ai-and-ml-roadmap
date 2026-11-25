import os
import shutil

def on_pre_build(config):
    """Sync repository documentation tree into isolated .docs_build staging directory,
    preserving exact repository folder hierarchies so all relative links resolve cleanly.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staging_dir = os.path.join(repo_root, ".docs_build")

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 1. Copy directory trees preserving paths
    for folder in ["docs", "curriculum", "projects", "interview-kits", "resources", "assets"]:
        src = os.path.join(repo_root, folder)
        dst = os.path.join(staging_dir, folder)
        if os.path.exists(src):
            shutil.copytree(src, dst)

    # 2. Copy root documents
    for doc_file in ["README.md", "ROADMAP.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE"]:
        src = os.path.join(repo_root, doc_file)
        dst = os.path.join(staging_dir, doc_file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # 3. Create index.md at root of staging from README.md for MkDocs home
    readme_src = os.path.join(repo_root, "README.md")
    index_dst = os.path.join(staging_dir, "index.md")
    if os.path.isfile(readme_src):
        shutil.copy2(readme_src, index_dst)
