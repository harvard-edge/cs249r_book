import json
import os

from absl import app
import requests

CONTRIBUTORS_FILE = '.all-contributorsrc'

EXCLUDED_USERS = {'web-flow', 'github-actions[bot]', 'mrdragonbear', 'jveejay'}

OWNER = "harvard-edge"
REPO = "cs249r_book"
BRANCH = "main"


def main(_):
    token = os.environ["GH_TOKEN"]

    headers = {
        "Authorization": f"token {token}"
    }

    web_address = f'https://api.github.com/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=100'
    res = requests.get(web_address, headers=headers)

    print(web_address)
    user_to_name_dict = dict()
    users_from_api = []

    if res.status_code == 200:
        data = res.json()

        for node in data:
            user_full_name = None
            username = None
            if node['commit']:
                commit = node['commit']
                if commit['author']:
                    author = commit['author']
                    user_full_name = author['name']
                elif commit['committer']:
                    committer = commit['committer']
                    user_full_name = committer['name']
            if node['committer']:
                committer = node['committer']
                username = committer['login']
            if node['author']:
                author = node['author']
                if author['login']:
                    username = author['login']

            assert user_full_name is not None, 'User full name should not be None'
            assert username is not None, 'Username should not be None'

            user_to_name_dict[username] = user_full_name
            users_from_api.append(username)

        users_from_api = set(users_from_api)
        print('Users pulled from API: ', users_from_api)

        with open(CONTRIBUTORS_FILE, 'r') as contrib_file:
            existing_contributor_data = json.load(contrib_file)
            existing_contributors = existing_contributor_data['contributors']

            existing_contributor_logins = []
            for existing_contributor in existing_contributors:
                user_to_name_dict[existing_contributor['login']] = existing_contributor['name']
                existing_contributor_logins.append(existing_contributor['login'])
            existing_contributor_logins_set = set(existing_contributor_logins)
            print('Existing contributors: ', existing_contributor_logins_set)
            existing_contributor_logins_set -= EXCLUDED_USERS
            # All contributors in the file should be in the API
            assert existing_contributor_logins_set.issubset(
                users_from_api), 'All contributors in the .all-contributorsrc file should be pulled using the API'

            new_contributor_logins = users_from_api - existing_contributor_logins_set
            print('New contributors: ', new_contributor_logins)

            result = users_from_api - EXCLUDED_USERS

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
