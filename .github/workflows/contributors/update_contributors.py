import collections
import json
import os

from absl import app
import requests

CONTRIBUTORS_FILE = ".all-contributorsrc"

EXCLUDED_USERS = {"web-flow", "github-actions[bot]", "mrdragonbear", "jveejay"}

OWNER = "harvard-edge"
REPO = "cs249r_book"
BRANCH = "main"


def main(_):
    token = os.environ["GH_TOKEN"]

    headers = {"Authorization": f"token {token}"}

    data = []
    next_page = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=100"
    )
    last_page = None
    while next_page != last_page:
        print(f"Fetching page: {next_page}")
        res = requests.get(next_page, headers=headers)
        data.extend(res.json())
        next_page = res.links.get("next", {}).get("url", None)
        last_page = res.links.get("last", {}).get("url", None)

    user_to_name_dict = dict()
    name_to_user_dict = dict()
    users_from_api = set()
    user_full_names_from_api = set()

    for node in data:
        commit_info = node.get("commit", None)
        commit_author_info = commit_info.get("author", None)
        commit_commiter_info = commit_info.get("committer", None)
        author_info = node.get("author", None)
        committer_info = node.get("committer", None)
        committer_login_info = (
            committer_info.get("login", None) if committer_info else None
        )
        user_full_name = None
        username = None

        if commit_author_info:
            user_full_name = commit_author_info["name"]
        elif commit_commiter_info:
            user_full_name = commit_commiter_info["name"]

        if author_info:
            username = author_info["login"]
        elif committer_login_info:
            username = committer_login_info["login"]

        if user_full_name:
            name_to_user_dict[user_full_name] = username if username else None
            user_full_names_from_api.add(user_full_name)
        if username:
            user_to_name_dict[username] = user_full_name if user_full_name else None
            users_from_api.add(username)

    print("Users pulled from API: ", users_from_api)

    with open(CONTRIBUTORS_FILE, "r") as contrib_file:
        existing_contributor_data = json.load(contrib_file)
        existing_contributors = existing_contributor_data["contributors"]

        existing_contributor_logins = []
        for existing_contributor in existing_contributors:
            user_to_name_dict[existing_contributor["login"]] = existing_contributor[
                "name"
            ]
            existing_contributor_logins.append(existing_contributor["login"])
        existing_contributor_logins_set = set(existing_contributor_logins)
        print("Existing contributors: ", existing_contributor_logins_set)
        existing_contributor_logins_set -= EXCLUDED_USERS
        # All contributors in the file should be in the API
        assert existing_contributor_logins_set.issubset(
            users_from_api
        ), "All contributors in the .all-contributorsrc file should be pulled using the API"

        new_contributor_logins = users_from_api - existing_contributor_logins_set
        print("New contributors: ", new_contributor_logins - EXCLUDED_USERS)

        result = users_from_api - EXCLUDED_USERS

        final_result = dict(
            projectName=REPO,
            projectOwner=OWNER,
            files=["contributors.qmd", "README.md"],
            contributors=[
                dict(
                    login=user,
                    name=user_to_name_dict[user] or user,
                    # If the user has no full name listed, use their username
                    avatar_url=f"https://avatars.githubusercontent.com/{user}",
                    profile=f"https://github.com/{user}",
                    # contributions=["doc"],
                    contributions=[],
                )
                for user in result
            ],
            repoType="github",
            contributorsPerLine=5,
            repoHost="https=//github.com",
            commitConvention="angular",
            skipCi=True,
            # commitType="docs"
        )

        print(final_result)
        json_string = json.dumps(
            final_result, indent=4
        )  # The indent parameter is optional, but it formats the output to be more readable
        print(json_string)

    with open(CONTRIBUTORS_FILE, "w") as contrib_file:
        contrib_file.write(json_string)


if __name__ == "__main__":
    app.run(main)
