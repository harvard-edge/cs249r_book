import hashlib
import json
import os
import random

import pandas as pd
import requests
from absl import app
from absl import logging

CONTRIBUTORS_FILE = ".all-contributorsrc"

EXCLUDED_USERS = {"web-flow", "github-actions[bot]", "mrdragonbear", "jveejay",
                  "Matthew Steward"}

OWNER = "harvard-edge"
REPO = "cs249r_book"
BRANCH = "main"


def get_user_data_from_username(username):
  headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
  res = requests.get(f"https://api.github.com/users/{username}",
                     headers=headers)
  user_full_name = pd.NA
  email_address = pd.NA
  if res.status_code == 200:
    user_data = res.json()
    user_full_name = user_data['name']
    email_address = user_data['email']
  else:
    logging.error(f'Could not find user with username: {username}')
  return {'username': username, 'user_full_name': user_full_name,
          'email_address': email_address}


def get_user_data_from_email(email_address):
  headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
  res = requests.get(f"https://api.github.com/search/users?q={email_address}",
                     headers=headers)
  username = pd.NA
  if res.status_code == 200:
    user_data = res.json()
    if user_data['total_count'] > 0:
      username = user_data['items'][0]['login']
  else:
    logging.error(f'Could not find user with email address: {email_address}')
  return {'username': username, 'user_full_name': pd.NA,
          'email_address': email_address}


def get_co_authors_from_commit_message(commit_message):
  co_author_data = []
  if commit_message:
    lines = commit_message.splitlines()
    for line in lines:
      try:
        if line.startswith("Co-authored-by:"):
          co_author = line.split(":")[1].strip()
          user_full_name, email_address = co_author.split("<")
          user_full_name = user_full_name.strip()
          email_address = email_address.strip(">")
          co_author_data.append(
              {'user_full_name': user_full_name,
               'email_address': email_address})
      except ValueError as e:
        logging.error(
            f"Error parsing co-author: {line}. Co-author should be of the form: "
            f"'Co-authored-by: NAME <email>'. "
            f"Remember to include the angle brackets around the email."
        )
    return pd.DataFrame(co_author_data)


def main(_):
  token = os.environ["GITHUB_TOKEN"]
  headers = {"Authorization": f"token {token}"}
  data = []
  next_page = (
      f"https://api.github.com/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=500"
  )
  last_page = None
  while next_page != last_page:
    print(f"Fetching page: {next_page}")
    res = requests.get(next_page, headers=headers)
    data.extend(res.json())
    next_page = res.links.get("next", {}).get("url", None)
    last_page = res.links.get("last", {}).get("url", None)

  commit_data = []
  for node in data:
    commit_message = node.get("commit", {}).get("message", pd.NA)
    commit_info = node.get("commit", None)
    commit_author_info = commit_info.get("author", None)
    commit_commiter_info = commit_info.get("committer", None)
    author_info = node.get("author", None)
    committer_info = node.get("committer", None)
    committer_login_info = (
        committer_info.get("login", None) if committer_info else None
    )
    user_full_name = pd.NA
    username = pd.NA

    if commit_author_info:
      user_full_name = commit_author_info["name"]
    elif commit_commiter_info:
      user_full_name = commit_commiter_info["name"]

    if author_info:
      username = author_info["login"]
    elif committer_login_info:
      username = committer_login_info["login"]

    commit_data.append(
        {
            "commit_message": commit_message,
            "user_full_name": user_full_name,
            "username": username,
        }
    )
  commit_data_df = pd.DataFrame(commit_data)
  co_authors_list = [get_co_authors_from_commit_message(row["commit_message"])
                     for index, row in commit_data_df.iterrows()]
  co_authors_df = pd.concat(co_authors_list, ignore_index=True)
  co_authors_df.drop_duplicates(inplace=True)

  # Merge the co-authors with the commit data
  commit_data_df.drop(columns=["commit_message"], inplace=True)
  commit_data_df = commit_data_df.merge(
      co_authors_df,
      how='outer',
      on=['user_full_name', ])

  # Remove rows where the username or user_full_name is in the EXCLUDED_USERS list in one line
  commit_data_df = commit_data_df[
    ~commit_data_df['username'].isin(EXCLUDED_USERS)
    & ~commit_data_df['user_full_name'].isin(EXCLUDED_USERS)
    ]

  # Before we drop duplicates, get the number of commits per user
  commit_data_df = commit_data_df.assign(
      commits=commit_data_df['user_full_name'].map(
          commit_data_df['user_full_name'].value_counts()))
  commit_data_df.drop_duplicates(inplace=True)

  # Use the API to look up all user info
  for index, row in commit_data_df.iterrows():
    if not pd.isna(row.username):
      user_data = get_user_data_from_username(row.username)
      commit_data_df.loc[index, 'username'] = user_data['username']

      if pd.isna(row.user_full_name) or (
          row.user_full_name == row.username and not pd.isna(
          user_data['user_full_name'])):
        commit_data_df.loc[index, 'user_full_name'] = user_data[
          'user_full_name']
      if pd.isna(row.email_address):
        commit_data_df.loc[index, 'email_address'] = user_data['email_address']
    elif not pd.isna(row.email_address):
      user_data = get_user_data_from_email(row.email_address)
      commit_data_df.loc[index, 'email_address'] = user_data['email_address']

      if pd.isna(row.username):
        commit_data_df.loc[index, 'username'] = user_data['username']
      if pd.isna(row.user_full_name):
        commit_data_df.loc[index, 'user_full_name'] = user_data[
          'user_full_name']
    else:
      logging.error(f"Could not find user for row: {row}")
  commit_data_df.drop_duplicates(inplace=True)

  # Get name length to figure out which full name to use
  commit_data_df = commit_data_df.assign(
      name_length=commit_data_df['user_full_name'].str.len())
  commit_data_df = commit_data_df.fillna(pd.NA)
  commit_data_df = commit_data_df.sort_values(by=['commits', 'name_length'],
                                              ascending=False)

  # Add a flag column for whether 'username' is NaN
  commit_data_df['has_username'] = ~commit_data_df['username'].isna()

  # Multi-level group by 'has_username', 'username', and 'email_address'
  commit_data_df = commit_data_df.groupby(
      ['has_username', 'username', 'email_address'],
      dropna=False,
      as_index=False).first()

  # Drop the 'has_username' column as it's no longer needed after grouping
  commit_data_df.drop('has_username', axis=1, inplace=True)
  commit_data_df.drop('name_length', axis=1, inplace=True)

  # If the user_full_name is an email address, replace it with the username
  commit_data_df['user_full_name'] = commit_data_df.apply(
      lambda row: row['username'] if '@' in row['user_full_name'] else row[
        'user_full_name'],
      axis=1)

  def generate_gravatar_url(name):
    name_list = list(name)
    random.shuffle(name_list)
    name = ''.join(name_list)
    name_hash = hashlib.md5(name.encode('utf-8')).hexdigest()
    return f"https://www.gravatar.com/avatar/{name_hash}?d=identicon&s=100"

  # Update avatar_url
  commit_data_df['avatar_url'] = commit_data_df.apply(
      lambda row: generate_gravatar_url(row['user_full_name']) if pd.isna(row[
                                                                            'username']) else f"https://avatars.githubusercontent.com/{row['username']}",
      axis=1)

  # Update profile URL
  commit_data_df['profile'] = commit_data_df.apply(
      lambda
          row: "https://github.com/harvard-edge/cs249r_book/graphs/contributors" if pd.isna(
          row['username']) else f"https://github.com/{row['username']}",
      axis=1)

  # Sort by number of commits
  commit_data_df.sort_values(by='commits', ascending=False, inplace=True)

  final_result = dict(
      projectName=REPO,
      projectOwner=OWNER,
      files=["contents/contributors.qmd", "README.md"],
      contributors=[
          dict(
              login=row.username,
              name=row.user_full_name if not pd.isna(
                  row.user_full_name) else row.username,
              avatar_url=row.avatar_url,
              profile=row.profile,
              contributions=[],
          )
          for row in commit_data_df.itertuples()
      ],
      repoType="github",
      contributorsPerLine=5,
      repoHost="https://github.com",
      commitConvention="angular",
      skipCi=True,
  )

  # Now, you can use final_result as needed

  json_string = json.dumps(
      final_result, indent=4
  )
  print(json_string)

  with open(CONTRIBUTORS_FILE, "w") as contrib_file:
    contrib_file.write(json_string)


if __name__ == "__main__":
  app.run(main)
