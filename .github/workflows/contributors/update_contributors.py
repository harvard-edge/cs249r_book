import json
import os

from absl import app
from github import Github
import requests

CONTRIBUTORS_FILE = '.all-contributorsrc'

EXCLUDED_USERS = ['web-flow', 'github-actions[bot]', 'mrdragonbear']

OWNER = "harvard-edge"
REPO = "cs249r_book"
BRANCH = "main"


def split_name_email(s):
    parts = s.rsplit(' ', 1)
    return parts[0], parts[1][1:-1]  # Removing angle brackets from email


def get_github_username(token, email):
    g = Github(token)
    users = g.search_users(email)
    for user in users:
        # Assuming the first user returned with the matching email is the correct user
        return user.login
    return None


def main(_):
    token = os.environ["GH_TOKEN"]

    headers = {
        "Authorization": f"token {token}"
    }

    web_address = f'https://api.github.com/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=100'
    res = requests.get(web_address, headers=headers)

    print(web_address)

    # Check if the request was successful
    if res.status_code == 200:
        # Parse the JSON response
        data = res.json()

        # Extract the 'login' attribute for each committer
        usernames = [commit['committer']['login'] for commit in data if commit['committer']]

        # Print unique usernames
        for username in sorted(set(usernames)):
            print(username)

        with open(CONTRIBUTORS_FILE, 'r') as contrib_file:
            contributors_data = json.load(contrib_file)
            user_to_name_dict = dict()
            contributors = contributors_data['contributors']

            contributor_logins = []
            for contrib in contributors:
                user_to_name_dict[contrib['login']] = contrib['name']
                contributor_logins.append(contrib['login'])
            contributor_logins_set = set(contributor_logins)

            # Perform the set subtraction
            # result = usernames_set - contributor_logins_set
            result = contributor_logins_set - set(EXCLUDED_USERS)

            print('New contributors: ', result)

            final_result = dict(
                projectName=REPO,
                projectOwner=OWNER,
                files=["contributors.qmd", "README.md"],
                contributors=[dict(login=user,
                                   name=user_to_name_dict[user],
                                   avatar_url=f'https://avatars.githubusercontent.com/{user}',
                                   profile=f'https://github.com/{user}',
                                   contributions=['doc'], ) for
                              user in result],

                repoType='github',
                contributorsPerLine=7,
                repoHost="https=//github.com",
                commitConvention='angular',
                skipCi=True,
                commitType="docs"
            )

            print(final_result)
            json_string = json.dumps(final_result,
                                     indent=4)  # The indent parameter is optional, but it formats the output to be more readable
            print(json_string)

        with open(CONTRIBUTORS_FILE, 'w') as contrib_file:
            contrib_file.write(json_string)
    else:
        print(f"Failed with status code: {res.status_code}")


if __name__ == '__main__':
    app.run(main)
