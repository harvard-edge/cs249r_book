import json
import os

import numpy as np
import pandas as pd
import github
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
  g = github.Github(os.environ["GITHUB_TOKEN"])
  try:
    user = g.get_user(username)
    return {'username': user.login, 'user_full_name': user.name,
            'email_address': user.email}
  except github.GithubException:
    return {'username': username, 'user_full_name': pd.NA,
            'email_address': pd.NA}


def get_user_data_from_email(email_address):
  g = github.Github(os.environ["GITHUB_TOKEN"])
  try:
    user = g.get_user(email_address)
    return {'username': user.login, 'user_full_name': user.name,
            'email_address': user.email}
  except github.GithubException:
    return {'username': pd.NA, 'user_full_name': pd.NA,
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
      f"https://api.github.com/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=100"
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

  commit_data_df.drop(columns=["commit_message"], inplace=True)
  commit_data_df.drop_duplicates(inplace=True)
  commit_data_df = commit_data_df.merge(
      co_authors_df, how='outer', on='user_full_name')
  commit_data_df.drop_duplicates(inplace=True)

  # Remove rows with excluded users
  commit_data_df = commit_data_df[~commit_data_df["username"].isin(
      EXCLUDED_USERS)]
  commit_data_df = commit_data_df[~commit_data_df["user_full_name"].isin(
      EXCLUDED_USERS)]

  # Use API to get user data for any missing users
  for index, row in commit_data_df.iterrows():
    user_data = None
    if pd.isna(row.username) and not pd.isna(row.email_address):
      user_data = get_user_data_from_email(row.email_address)
      commit_data_df.at[index, 'username'] = user_data['username']
    elif not pd.isna(row.username) and pd.isna(row.email_address):
      user_data = get_user_data_from_username(row.username)
      commit_data_df.at[index, 'email_address'] = user_data['email_address']

    # Only replace user_full_name if it's missing, or if it's different from what we have
    if user_data and not pd.isna(user_data['user_full_name']):
      if pd.isna(row.user_full_name) or user_data[
        'user_full_name'] != row.user_full_name:
        commit_data_df.at[index, 'user_full_name'] = user_data['user_full_name']

  # Get name length to figure out which full name to use
  commit_data_df = commit_data_df.assign(
      name_length=commit_data_df['user_full_name'].str.len())
  commit_data_df = commit_data_df.fillna(pd.NA)
  commit_data_df = commit_data_df.sort_values(by='name_length', ascending=False)

  # Add a flag column for whether 'username' is NaN
  commit_data_df['has_username'] = ~commit_data_df['username'].isna()

  # Multi-level group by 'has_username' and 'username'
  commit_data_df = commit_data_df.groupby(
      ['has_username', 'username', 'email_address'],
      dropna=False,
      as_index=False).first()

  # Drop the 'has_username' column as it's no longer needed after grouping
  commit_data_df.drop('has_username', axis=1, inplace=True)
  commit_data_df['display_name'] = commit_data_df['username'].fillna('octocat')

  # Shuffle the DataFrame rows
  commit_data_df = commit_data_df.sample(frac=1).reset_index(drop=True)

  # Sort the DataFrame to put 'octocat' at the bottom
  commit_data_df['is_octocat'] = commit_data_df['display_name'] == 'octocat'
  commit_data_df = commit_data_df.sort_values(by='is_octocat',
                                              ascending=True).drop('is_octocat',
                                                                   axis=1)

  final_result = dict(
      projectName=REPO,
      projectOwner=OWNER,
      files=["contributors.qmd", "README.md"],
      contributors=[
          dict(
              login=row.display_name,
              name=row.user_full_name,
              avatar_url=f"https://avatars.githubusercontent.com/{row.display_name}",
              profile=f"https://github.com/{row.display_name}",
              contributions=[],
          )
          for row in commit_data_df.itertuples()
      ],
      repoType="github",
      contributorsPerLine=5,
      repoHost="https=//github.com",
      commitConvention="angular",
      skipCi=True,
  )

  json_string = json.dumps(
      final_result, indent=4
  )
  print(json_string)

  with open(CONTRIBUTORS_FILE, "w") as contrib_file:
    contrib_file.write(json_string)


if __name__ == "__main__":
  app.run(main)
